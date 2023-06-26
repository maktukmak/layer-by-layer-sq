import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
import time
from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert
import math
import re
from transformers import OPTForCausalLM, BloomForCausalLM
from tqdm import tqdm
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer
from transformers.models.bloom.modeling_bloom import BloomBlock
from contextlib import nullcontext


def create_dummy_past_key_val(model, batch_size, dummy_len = 1):
    model_type = str(model.__class__) 
    if re.search('opt', model_type):
        past_key_values = [(torch.zeros(batch_size, model.config.num_attention_heads, dummy_len, model.config.hidden_size // model.config.num_attention_heads),)*2] * model.config.num_hidden_layers
    if re.search('bloom', model_type):
        past_key_values = [(torch.zeros(batch_size * model.config.num_attention_heads, model.config.hidden_size // model.config.num_attention_heads, dummy_len),
                                torch.zeros(batch_size * model.config.num_attention_heads, dummy_len, model.config.hidden_size // model.config.num_attention_heads))] * model.config.num_hidden_layers
    return past_key_values

def remove_quant_layers(model, user_model, forward_org):
    user_model.lm_head = None
    if re.search('opt', model):
        for i in range(len(user_model.model.decoder.layers)):
            user_model.model.decoder.layers[i] = None
        user_model.model.decoder.forward = forward_org
    if re.search('bloom', model):
        for i in range(len(user_model.transformer.h)):
            user_model.transformer.h[i] = None
        user_model.transformer.forward = forward_org
        #user_model.to(torch.bfloat16)

def hook_activations(model, layers, lm_head, calib_inp, dict_inp_act, q_lm_head, q_layers):

    model_name = str(model.__class__) 
    def get_activation(name):
        def hook(model, input, output):
            inputs = (input[0].detach().clone(),) # hidden states
            if not name == 'lm_head':
                if  re.search('opt', model_name):
                    inputs += (input[1].detach().clone(),) # attention mask
                    inputs += (tuple(i.detach().clone() for i in input[2]),) # past_key_val 
                elif re.search('bloom', model_name):
                    inputs += (input[1].detach().clone(),) # alibi, 
                    inputs += (input[2].detach().clone(),) # attention mask
                    inputs += (tuple(i.detach().clone() for i in input[3]),) # past_key_val 
            dict_inp_act[name].append(inputs)
        return hook

    hooks = [layers[i].register_forward_hook(get_activation('layer' + str(i))) for i in q_layers]
    if q_lm_head:
        hook_lm_head = lm_head.register_forward_hook(get_activation('lm_head'))

    # Dummy past_key_val for jit tracing
    batch_size = calib_inp[0].shape[0]
    calib_seq_len = calib_inp[0].shape[1]

    mask = pad(torch.ones((batch_size, calib_seq_len)), (1, 0), value=0)
    past_key_values = create_dummy_past_key_val(model, batch_size)

    for inp in calib_inp:
        with torch.no_grad():
            model(  inp,
                    attention_mask = mask,
                    past_key_values = past_key_values)

    [h.remove() for h in hooks]
    hook_lm_head.remove()

def quantize_module(layer, calib_inp, qconfig, inplace=False, sq = False, sq_auto = False, alpha = 0.5, amp = False):

    '''
        layer: pytorch module
        inp: batch x sample x (tuple)
    '''
    layer_type = str(layer.__class__)
    if re.search('opt', layer_type):
        fn_block_convert_to_smooth_quant = opt_block_convert_to_smooth_quant
        model_class = OPTDecoderLayer
    if re.search('bloom', layer_type):
        fn_block_convert_to_smooth_quant = bloom_block_convert_to_smooth_quant
        model_class = BloomBlock

    def block_prepare_smooth_quant_wrapper(layer, alpha):
        layer_sq = prepare(layer, qconfig_sq, example_inputs=calib_inp[0], inplace=False)
        [layer_sq(*inp) for inp in calib_inp]
        fn_block_convert_to_smooth_quant(' ', layer_sq, layer_sq, alpha = alpha)
        layer_sq.__class__ = model_class
        del layer_sq._fqn_to_auto_quant_state_map
        return layer_sq

    if sq:
        qconfig_sq = QConfig(activation=PerChannelMinMaxObserver.with_args(ch_axis = -1, qscheme=torch.torch.per_channel_affine, dtype=torch.quint8),
                    weight=PerChannelMinMaxObserver.with_args(ch_axis = -1, dtype=torch.qint8, qscheme=torch.torch.per_channel_affine,))
        ind = 0
        rng = [0.5]
        if sq_auto:
            sq_rng = torch.arange(0, 1 + 0.2, 0.2)
            err_list = []
            for alpha in sq_rng:

                layer_sq = block_prepare_smooth_quant_wrapper(layer, alpha)

                prepare(layer_sq, qconfig, example_inputs=calib_inp[0], inplace=True)
                [layer_sq(*inp) for inp in calib_inp]
                convert(layer_sq, inplace=True)

                #with torch.no_grad():
                #    with torch.cpu.amp.autocast() if amp else nullcontext():
                #        layer_sq = torch.jit.trace(layer_sq, calib_inp[0], check_trace=False) # Consumes a lot of memory when check_trace 
                #        layer_sq = torch.jit.freeze(layer_sq)
                #print(alpha)
                err_list.append(torch.sqrt(torch.mean((layer(*calib_inp[0])[0] - layer_sq(*calib_inp[0])[0])**2)).detach())
            print(err_list)
            ind = torch.argmin(torch.Tensor(err_list))
            print('Selected alpha:', sq_rng[ind])

        layer = block_prepare_smooth_quant_wrapper(layer, sq_rng[ind])

    if inplace:
        prepare(layer, qconfig, example_inputs=calib_inp[0], inplace=inplace)
    else:
        layer = prepare(layer, qconfig, example_inputs=calib_inp[0], inplace=inplace)

    [layer(*inp) for inp in calib_inp]

    convert(layer, inplace=True)

    with torch.no_grad():
        with torch.cpu.amp.autocast() if amp else nullcontext():
            layer = torch.jit.trace(layer, calib_inp[0], check_trace=False) # Consumes a lot of memory when check_trace 
            layer = torch.jit.freeze(layer)

    return layer


def bloom_block_convert_to_smooth_quant(block_name, block, prepared_model, alpha):

    qmap = prepared_model._fqn_to_auto_quant_state_map

    for op in qmap[block_name].seen_nonq_op_infos:
        if re.search('input_layernorm', op.fqn):
            input_layer_norm = op.output_tensor_infos[0].id
        if re.search('post_attention_layernorm', op.fqn):
            post_att_layer_norm = op.output_tensor_infos[0].id

    if block_name == ' ':
        q_name_attn = 'self_attention'
        q_name_mlp = 'mlp'
    else:
        q_name_attn = block_name + ':self_attn' 
        q_name_mlp = block_name + ':mlp'

    # Input layer norm scale
    min_vec_a = qmap[q_name_attn].tensor_id_to_observer[str(input_layer_norm)].min_val
    max_vec_a = qmap[q_name_attn].tensor_id_to_observer[str(input_layer_norm)].max_val
    scale_a_inp_layer_norm = torch.max(torch.vstack([abs(min_vec_a), abs(max_vec_a)]), axis = 0).values 

    # Attn linear layers scale
    min_vec_w = qmap[q_name_attn].weight_tensor_id_to_observer['0_0'].min_val
    max_vec_w = qmap[q_name_attn].weight_tensor_id_to_observer['0_0'].max_val
    scale_w_attn = torch.max(torch.vstack([abs(min_vec_w), abs(max_vec_w)]), axis = 0).values 


    # Post attention layer norm scale
    min_vec_a = qmap[q_name_mlp].tensor_id_to_observer[str(post_att_layer_norm)].min_val
    max_vec_a = qmap[q_name_mlp].tensor_id_to_observer[str(post_att_layer_norm)].max_val
    scale_a_post_attn_layer_norm = torch.max(torch.vstack([abs(min_vec_a), abs(max_vec_a)]), axis = 0).values

    # Fc1 scale
    min_vec_w = qmap[q_name_mlp].weight_tensor_id_to_observer['0_0'].min_val
    max_vec_w = qmap[q_name_mlp].weight_tensor_id_to_observer['0_0'].max_val
    scale_w_fc1 = torch.max(torch.vstack([abs(min_vec_w), abs(max_vec_w)]), axis = 0).values


    s_w_attn = torch.pow(scale_a_inp_layer_norm, alpha) / torch.pow(scale_w_attn, 1-alpha)
    s_attn_ln = s_w_attn
    s_w_fc1 = torch.pow(scale_a_post_attn_layer_norm, alpha) / torch.pow(scale_w_fc1, 1-alpha)
    s_final_ln = s_w_fc1


    # Scale
    block.input_layernorm.state_dict()['weight'] /= s_attn_ln
    block.input_layernorm.state_dict()['bias'] /= s_attn_ln
    block.self_attention.query_key_value.state_dict()['weight'] *= s_w_attn[None, :]

    block.post_attention_layernorm.state_dict()['weight'] /= s_final_ln
    block.post_attention_layernorm.state_dict()['bias'] /= s_final_ln
    block.mlp.dense_h_to_4h.state_dict()['weight'] *= s_w_fc1[None, :]


def bloom_convert_to_smooth_quant(model, calib_inputs, alpha = 0.5):

    qconfig = QConfig(activation=PerChannelMinMaxObserver.with_args(ch_axis = -1, qscheme=torch.torch.per_channel_affine, dtype=torch.quint8),
                    weight=PerChannelMinMaxObserver.with_args(ch_axis = -1, dtype=torch.qint8, qscheme=torch.torch.per_channel_affine,))
    prepare(model, qconfig, example_inputs=calib_inputs, inplace=True)
    model(calib_inputs)

    for i in range(model.config.n_layer):
        bloom_block_convert_to_smooth_quant('transformer:h:' + str(i), model.transformer.h[i], model, alpha)

    # Final quant
    qconfig = QConfig(activation=MinMaxObserver.with_args(qscheme=torch.torch.per_tensor_affine, dtype=torch.quint8),
                    weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.torch.per_tensor_affine,))
    del model._fqn_to_auto_quant_state_map
    model.__class__ = BloomForCausalLM
    prepare(model, qconfig, example_inputs=calib_inputs, inplace=True)
    model(calib_inputs)

def opt_block_convert_to_smooth_quant(block_name, block, prepared_model, alpha):

    qmap = prepared_model._fqn_to_auto_quant_state_map

    for op in qmap[block_name].seen_nonq_op_infos:
        if re.search('self_attn_layer_norm', op.fqn):
            id_attn_layer_norm = op.output_tensor_infos[0].id
        if re.search('final_layer_norm', op.fqn):
            id_final_layer_norm = op.output_tensor_infos[0].id

    if block_name == ' ':
        q_name = 'self_attn'
    else:
        q_name = block_name + ':self_attn'

    # Attn_layer_norm scale
    min_vec_a = qmap[q_name].tensor_id_to_observer[str(id_attn_layer_norm)].min_val
    max_vec_a = qmap[q_name].tensor_id_to_observer[str(id_attn_layer_norm)].max_val
    scale_a_attn_layer_norm = torch.max(torch.vstack([abs(min_vec_a), abs(max_vec_a)]), axis = 0).values 
    
    # Attn linear layers scale
    min_vec_w = torch.vstack([qmap[q_name].weight_tensor_id_to_observer[k].min_val for k in ['0_0', '1_0', '2_0']])
    max_vec_w = torch.vstack([qmap[q_name].weight_tensor_id_to_observer[k].max_val for k in ['0_0', '1_0', '2_0']])
    scale_w_attn = torch.max(torch.vstack([abs(min_vec_w), abs(max_vec_w)]), axis = 0).values 

    # Final_layer_norm scale
    min_vec_a = qmap[block_name].tensor_id_to_observer[str(id_final_layer_norm)].min_val
    max_vec_a = qmap[block_name].tensor_id_to_observer[str(id_final_layer_norm)].max_val
    scale_a_final_layer_norm = torch.max(torch.vstack([abs(min_vec_a), abs(max_vec_a)]), axis = 0).values

    # Fc1 scale
    min_vec_w = qmap[block_name].weight_tensor_id_to_observer['1_0'].min_val
    max_vec_w = qmap[block_name].weight_tensor_id_to_observer['1_0'].max_val
    scale_w_fc1 = torch.max(torch.vstack([abs(min_vec_w), abs(max_vec_w)]), axis = 0).values


    s_w_attn = torch.pow(scale_a_attn_layer_norm, alpha) / torch.pow(scale_w_attn, 1-alpha)
    s_attn_ln = s_w_attn
    s_w_fc1 = torch.pow(scale_a_final_layer_norm, alpha) / torch.pow(scale_w_fc1, 1-alpha)
    s_final_ln = s_w_fc1
    
    # Scale
    block.self_attn_layer_norm.state_dict()['weight'] /= s_attn_ln
    block.self_attn_layer_norm.state_dict()['bias'] /= s_attn_ln
    block.fc1.state_dict()['weight'] *= s_w_fc1[None, :]

    block.self_attn.q_proj.state_dict()['weight'] *= s_w_attn[None, :]
    block.self_attn.k_proj.state_dict()['weight'] *= s_w_attn[None, :]
    block.self_attn.v_proj.state_dict()['weight'] *= s_w_attn[None, :]

    block.final_layer_norm.state_dict()['weight'] /= s_final_ln
    block.final_layer_norm.state_dict()['bias'] /= s_final_ln

def opt_convert_to_smooth_quant(model, calib_inputs, alpha = 0.5):

    # Initial quant module for scaling
    qconfig = QConfig(activation=PerChannelMinMaxObserver.with_args(ch_axis = -1, qscheme=torch.torch.per_channel_affine, dtype=torch.quint8),
                    weight=PerChannelMinMaxObserver.with_args(ch_axis = -1, dtype=torch.qint8, qscheme=torch.torch.per_channel_affine,))
    prepare(model, qconfig, example_inputs=calib_inputs[0], inplace=True)
    for inp in calib_inputs:
        model(*inp)

    for i in range(len(model.model.decoder.layers)):
        opt_block_convert_to_smooth_quant('model:decoder:layers:' + str(i), model.model.decoder.layers[i], model, alpha)

    del model._fqn_to_auto_quant_state_map
    model.__class__ = OPTForCausalLM
    # Final quant
    #qconfig = QConfig(activation=MinMaxObserver.with_args(qscheme=torch.torch.per_tensor_affine, dtype=torch.quint8),
    #                weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.torch.per_tensor_affine,))
    #prepare(model, qconfig, example_inputs=calib_inputs, inplace=True)
    #model(calib_inputs)

class Evaluator:
    def __init__(self, dataset, tokenizer, batch_size = 8, pad_val = 1, pad_max = 512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.pad_val = pad_val
        self.pad_max = pad_max

        # tokenize the dataset
        self.dataset = self.dataset.map(self.tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def tokenize_function(self, examples):
        example = self.tokenizer(examples['text'])
        return example

    @torch.no_grad()
    def collate_batch(self, batch):
   
        input_ids_padded = []
        last_ind = []
        
        for text in batch:
            input_ids = text['input_ids']
            pad_len = self.pad_max - input_ids.shape[0]
            last_ind.append(input_ids.shape[0]-1)

            input_ids = pad(input_ids, (0, pad_len), value=self.pad_val)
            input_ids_padded.append(input_ids)
            
        
        return (torch.vstack(input_ids_padded), torch.tensor(last_ind))

    
    #@torch.no_grad()
    #@profile
    def evaluate(self, model):

        #model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        latency = 0
        test_dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_batch)
        for i, (input_ids, last_ind) in enumerate(test_dataloader):

            label = input_ids[torch.arange(len(last_ind)), last_ind]
            input_ids[torch.arange(len(last_ind)), last_ind] = self.pad_val
            pad_len = self.pad_max - last_ind - 1 

            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            mask = pad(torch.ones((batch_size, seq_len)), (1, 0), value=0)
            past_key_values = create_dummy_past_key_val(model, batch_size)

            start = time.time()
            with torch.no_grad():
                outputs = model(input_ids, 
                                attention_mask = mask,
                                past_key_values = past_key_values)
            latency += (time.time() - start)

            last_token_logits = outputs[0][torch.arange(len(last_ind)), -2-pad_len, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            if (i % 10 == 0):
                print('Processed minibatch:', i)

        acc = hit / total
        lantecy = latency / len(self.dataset)
        return acc, lantecy



    def evaluate_perp(self, model):

        encodings = self.tokenizer("\n\n".join(self.dataset["text"]), return_tensors="pt")
        stride = self.pad_max//4
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        input_ids_batch = []
        target_ids_batch = []
        i = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + self.pad_max, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            if end_loc - begin_loc < self.pad_max:
                input_ids = pad(input_ids, (0, self.pad_max - input_ids.shape[1]), value=self.pad_val)
                target_ids = pad(target_ids, (0, self.pad_max - target_ids.shape[1]), value=-100)

            input_ids_batch.append(input_ids)
            target_ids_batch.append(target_ids)

            i += 1
            if i == self.batch_size or end_loc == seq_len:

                mask = pad(torch.ones((i, input_ids_batch[0].shape[1])), (1, 0), value=0)
                past_key_values = create_dummy_past_key_val(model, i)

                with torch.no_grad():
                    outputs = model(torch.vstack(input_ids_batch),
                                    attention_mask = mask,
                                    past_key_values = past_key_values,
                                    labels=torch.vstack(target_ids_batch))
                    neg_log_likelihood = outputs[0] * trg_len * i
                
                i = 0
                input_ids_batch = []
                target_ids_batch = []
                nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        return ppl


# Add quant/dequant ops
def replace_layer(module):
    if isinstance(module, torch.nn.Linear):
        #target_state_dict   = deepcopy(module.state_dict())
        #bias                = True if module.bias is not None else False
        new_module = torch.nn.Sequential(
                    torch.quantization.QuantStub(), 
                    module, 
                    torch.quantization.DeQuantStub())
        #new_module.load_state_dict(target_state_dict)
        return new_module

def quant_tensor(W):
    s = (torch.max(W) - torch.min(W)) / 255
    z = -((torch.max(W) + torch.min(W)) / (2*s)).int()
    Wq = torch.quantize_per_tensor(W, s.item(), z.item(), torch.qint8)
    return Wq

def quant_channel(W, axis):
    axis = 0
    s = (torch.max(W, 1-axis).values - torch.min(W, 1-axis).values) / 255
    z = -((torch.max(W, 1-axis).values + torch.min(W, 1-axis).values) / (2*s)).int()
    Wq = torch.quantize_per_channel(W, s, z, axis, torch.qint8)
    return Wq