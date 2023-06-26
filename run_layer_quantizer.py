import os
import psutil
os.environ['DNNL_GRAPH_VERBOSE'] = '0'
import argparse
import pickle
import re
from collections import defaultdict
from types import MethodType
import torch
from intel_extension_for_pytorch.quantization import convert, prepare
from torch.ao.quantization import (MinMaxObserver, PerChannelMinMaxObserver,
                                   QConfig)
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, BloomForCausalLM, GPT2Tokenizer,
                          OPTForCausalLM)
from transformers.models.opt.modeling_opt import OPTAttention

from datasets import load_dataset, load_from_disk
from model_changes import (forward_bloom_decoder_mod,
                           forward_bloom_decoder_layer_mod,
                           forward_opt_decoder_layer_mod,
                           forward_opt_decoder_mod)
from utils import (Evaluator, hook_activations, opt_convert_to_smooth_quant,
                   quantize_module, remove_quant_layers,
                   create_dummy_past_key_val)
from torch.nn.functional import pad
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--method', nargs='?', default='sint8', const='sint8', help = 'sint8 (smoothquant) or int8 (conventional)')
parser.add_argument('--model', nargs='?', default='bigscience/bloom-560m', const='facebook/opt-125m', help = "Model selection, e.g., facebook/opt-125m, bigscience/bloom-560m")
parser.add_argument('--dataset_path', nargs='?', default='../datasets/', const='lambada', help = "For offline storing")
parser.add_argument('--dataset', nargs='?', default='lambada', const='lambada')
parser.add_argument('--split', nargs='?', default='validation', const='validation')
parser.add_argument('--calib_bathces', default=4, type=int, help = 'Number of minibatches used for calibration')
parser.add_argument('--batch_size', default=8, type=int, help = 'Number of samples in each batch')
parser.add_argument('--use_cache', default=True, action='store_true')
parser.add_argument('--sq_alpha', default=0.25, type=float, help = 'Smoothquant hyperparameter')
parser.add_argument('--sq_auto', default=True, action='store_true', help = 'Smoothquant hyperparameter automatic selection to minimize layer RMSE')
parser.add_argument('--calib_seq_len', default=512, type=int, help = 'Sequence length used for calibration and jit tracing')
parser.add_argument('--tmp_dir', nargs='?', default='../tmp/', const='../tmp/', help = "Model directory, ../tmp/ or /lfs/lfs12/maktukma/LLM/tmp/")
args = parser.parse_args()


save_dir = args.tmp_dir + 'quant_models/' + args.model.split("/")[1] + '/' + 'layer' + '_' + args.method + '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#Get process id
process = psutil.Process(os.getpid())
print("Initial: {}MB".format(int(process.memory_full_info().rss/(1024*1204))))

# Load the model
if re.search('opt', args.model):
    user_model = OPTForCausalLM.from_pretrained(args.model, torchscript=True, torch_dtype=torch.float32)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    layers = user_model.model.decoder.layers
    lm_head = user_model.lm_head
    decoder = user_model.model.decoder
    forward_mod = forward_opt_decoder_mod
    forward_layer_mod = forward_opt_decoder_layer_mod
    forward_org = decoder.forward
if re.search('bloom', args.model):
    user_model = BloomForCausalLM.from_pretrained(args.model, torchscript=True, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    layers = user_model.transformer.h
    lm_head = user_model.lm_head
    decoder = user_model.transformer
    forward_mod = forward_bloom_decoder_mod
    forward_layer_mod = forward_bloom_decoder_layer_mod
    forward_org = decoder.forward
print("Model and tokenizer Load: {}MB".format(int(process.memory_full_info().rss/(1024*1204))))

# Load the calibration dataset
if os.environ.get('HF_DATASETS_OFFLINE') == "1":
    dataset = load_from_disk(args.dataset_path + args.dataset)
    print('offline')
else:
    dataset = load_dataset(args.dataset, split=args.split)
    dataset.save_to_disk(args.dataset_path + args.dataset)
evaluator = Evaluator(dataset, tokenizer, args.batch_size)
calib_dataloader = DataLoader(evaluator.dataset, batch_size=args.batch_size, shuffle=False, collate_fn=evaluator.collate_batch)
calib_inp = []
for i, (input_ids, last_ind) in enumerate(calib_dataloader):
    calib_inp.append(input_ids[:,0:args.calib_seq_len])
    if i == args.calib_bathces-1:
        break
print("Dataset Load: {}MB".format(int(process.memory_full_info().rss/(1024*1204))))


# Model modification for layerwise quant (This is necessary because torch hooks don't capture kwargs (https://github.com/pytorch/pytorch/issues/35643))
user_model.to(torch.float32)
user_model.eval()
L = len(layers)
decoder.forward = MethodType(forward_mod, decoder)
for i in range(L):
    layers[i].forward = MethodType(forward_layer_mod, layers[i])
user_model.config.use_cache = args.use_cache

# Select layers to be quantized
q_lm_head = True
q_layers = range(L-1, -1, -1)

# Activation hooking at generation step
inps_cp = defaultdict(list)
hook_activations(user_model, layers, lm_head, calib_inp, inps_cp, q_lm_head, q_layers)
print("Hooking: {}MB".format(int(process.memory_full_info().rss/(1024*1204))))

# Layer by layer int8 trace
qconfig = QConfig(activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
                weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine))

if q_lm_head:
    traced_module = quantize_module(lm_head, inps_cp['lm_head'], qconfig, inplace = False, sq = False, amp = False)
    traced_module.save(save_dir + 'lm_head.pt')
    torch._C._jit_clear_class_registry()
    print('lm_head traced: {}MB'.format(int(process.memory_full_info().rss/(1024*1204))))

for i in q_layers:
    layer_name = 'layer' + str(i)
    traced_module = quantize_module(layers[i], inps_cp['layer' + str(i)], qconfig, inplace = False, sq = (args.method == 'sint8'), sq_auto = args.sq_auto, alpha = args.sq_alpha, amp = False)
    traced_module.save(save_dir + 'layer' + str(i) + '.pt')
    print("Layer {} traced: {}MB".format(int(i), int(process.memory_full_info().rss/(1024*1204))))
    print('Data types of the traced model:', set([traced_module._c.code_with_constants[1][t].dtype for t in traced_module._c.code_with_constants[1].keys() if torch.is_tensor(traced_module._c.code_with_constants[1][t])]))
    torch._C._jit_clear_class_registry()

remove_quant_layers(args.model, user_model, forward_org)
torch.save(user_model, save_dir + 'model_arch' + '.pt')
print('Model arch saved: {}MB'.format(int(process.memory_full_info().rss/(1024*1204))))
print('Quantization and tracing done')
