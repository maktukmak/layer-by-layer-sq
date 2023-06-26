import os
import psutil
os.environ['DNNL_GRAPH_VERBOSE'] = '0'
import argparse
import re
from types import MethodType
import intel_extension_for_pytorch as ipex
import torch
from intel_extension_for_pytorch.quantization import convert, prepare
from torch.ao.quantization import (MinMaxObserver, PerChannelMinMaxObserver,
                                   QConfig)
from torch.nn.functional import pad
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoTokenizer, BloomForCausalLM, GenerationConfig,
                          GPT2Tokenizer, OPTForCausalLM)
from datasets import load_dataset, load_from_disk
from utils import Evaluator
from model_changes import (forward_bloom_decoder_mod, forward_opt_decoder_mod,
                           bloom_prepare_inputs_for_generation,
                           opt_prepare_inputs_for_generation)

parser = argparse.ArgumentParser()
parser.add_argument('--method', nargs='?', default='sint8', const='sint8', help = 'org (fp32), sint8 (smoothquant) or int8 (conventional)')
parser.add_argument('--model', nargs='?', default='bigscience/bloom-560m', const='facebook/opt-125m', help = "Model selection, e.g., facebook/opt-125m, bigscience/bloom-560m")
parser.add_argument('--dataset_path', nargs='?', default='../datasets/', const='lambada', help = "For offline storing")
parser.add_argument('--dataset', nargs='?', default='lambada', const='lambada')
parser.add_argument('--split', nargs='?', default='validation[:1000]', const='validation')
parser.add_argument('--calib_bathces', default=4, type=int, help = 'Number of minibatches used for calibration')
parser.add_argument('--batch_size', default=8, type=int, help = 'Number of samples in each batch')
parser.add_argument('--use_cache', default = True, action='store_true')
parser.add_argument('--calib_seq_len', default=16, type=int, help = 'Sequence length used for calibration and jit tracing')
parser.add_argument('--tmp_dir', nargs='?', default='../tmp/', const='../tmp/', help = "Model directory, ../tmp/ or /lfs/lfs12/maktukma/LLM/tmp/")
parser.add_argument('--profile_perf', default = False, action='store_true')
parser.add_argument('--evaluate_acc', default = False, action='store_true')
parser.add_argument('--evaluate_perp', default = True, action='store_true')
parser.add_argument('--max_seq_len', default=512, type=int, help = 'Used to pad input text when evaluating acc/perp')
parser.add_argument('--generate', default = False, action='store_true')
args = parser.parse_args()


load_dir = args.tmp_dir + 'quant_models/' + args.model.split("/")[1] + '/' + 'layer' + '_' + args.method + '/'

#Get process id
process = psutil.Process(os.getpid())
print("Initial: {}MB".format(int(process.memory_full_info().rss/(1024*1204))))

tokenizer = AutoTokenizer.from_pretrained(args.model)
if os.environ.get('HF_DATASETS_OFFLINE') == "1":
    dataset = load_from_disk(args.dataset_path + args.dataset)
    print('offline')
else:
    dataset = load_dataset(args.dataset, split=args.split)
    dataset.save_to_disk(args.dataset_path + args.dataset)
print("Dataset Load: {}MB".format(int(process.memory_full_info().rss/(1024*1204))))

# Model load
if args.method == 'org':
    # Load the full model
    if re.search('opt', args.model):
        user_model = OPTForCausalLM.from_pretrained(args.model, torchscript=True, torch_dtype=torch.float32)
        tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    if re.search('bloom', args.model):
        user_model = BloomForCausalLM.from_pretrained(args.model, torchscript=True, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(args.model)

elif args.method == 'sint8' or args.method == 'int8':

    #Load model architecture
    user_model = torch.load(load_dir + 'model_arch.pt')
    if re.search('opt', args.model):
        layers = user_model.model.decoder.layers
        lm_head = user_model.lm_head
        decoder = user_model.model.decoder
        forward_mod = forward_opt_decoder_mod
        gen_input_fn = opt_prepare_inputs_for_generation
    if re.search('bloom', args.model):
        layers = user_model.transformer.h
        lm_head = user_model.lm_head
        decoder = user_model.transformer
        forward_mod = forward_bloom_decoder_mod
        gen_input_fn = bloom_prepare_inputs_for_generation
    print("Model arch loaded: {}MB".format(int(process.memory_full_info().rss/(1024*1204))))

    user_model.config.use_cache = args.use_cache
    user_model.eval()
    L = len(layers)

    # Model modification for layerwise quant (This is necessary because torch hooks don't capture kwargs (https://github.com/pytorch/pytorch/issues/35643))
    decoder.forward = MethodType(forward_mod, decoder)

    # This is for avoiding None valued past_key_vals (Adds a dummy token to the beginning)
    user_model.prepare_inputs_for_generation = MethodType(gen_input_fn, user_model)

    # Model stitch
    for i in range(L):
        traced_layer = torch.load(load_dir + 'layer' + str(i) + '.pt')
        layers[i] = traced_layer
    traced_layer = torch.load(load_dir + 'lm_head.pt')
    user_model.lm_head = traced_layer
    print("Model stitched: {}MB".format(int(process.memory_full_info().rss/(1024*1204))))
    

user_model.to(torch.float32)
evaluator = Evaluator(dataset, tokenizer, args.batch_size, pad_max = args.max_seq_len)
dataset_loader = DataLoader(evaluator.dataset, batch_size=args.batch_size, shuffle=False, collate_fn=evaluator.collate_batch)

if args.evaluate_acc:

    acc_fp32, latency_fp32 = evaluator.evaluate(user_model)
    print('Accuracy:', acc_fp32)
    print('Latency (sec):', latency_fp32)

if args.profile_perf:
    wait = 1
    warmup = 1
    active = 5
    # Profile
    def trace_handler(p):
        output = p.key_averages().table(sort_by="cpu_time_total", row_limit=20)
        print(output)

    with profile(activities=[ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active),
            on_trace_ready=trace_handler,
            profile_memory=False, 
            with_stack = False,
            with_flops = False,
            with_modules = True,
            record_shapes=True
            ) as prof:

        for i, (input_ids, last_ind) in enumerate(dataset_loader):
            with torch.no_grad():
                user_model(input_ids)
            prof.step()
            if i == wait + warmup + active - 1:
                break
    print("Profiling: {}MB".format(int(process.memory_full_info().rss/(1024*1204))))


if args.generate:
    
    generation_config = GenerationConfig(
    use_cache = args.use_cache,
    num_beams=4,
    num_return_sequences=1, 
    early_stopping=True,
    do_sample = True,
    #temperature=0.7,
    max_new_tokens=150,
    top_k=50,
    top_p=0.85, 
    decoder_start_token_id=0,
    no_repeat_ngram_size=2,
    eos_token_id=user_model.config.eos_token_id,
    pad_token=user_model.config.pad_token_id,
    )

    prompts = [
        "Hey, are you conscious? Can you talk to me?",
        "I look forward to",
        'I enjoy walking with my cute dog',
        "I don't know about you, but there's only one thing I want to do after a long day of work",
        'In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.',
        'Miley Cyrus was caught shoplifting from Abercrombie and Fitch on Hollywood Boulevard today.',
        'Legolas and Gimli advanced on the orcs, raising their weapons with a harrowing war cry.',
        "For today’s homework assignment, please describe the reasons for the US Civil War."
    ]


    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        generate_ids, _ = user_model.generate(**inputs, generation_config=generation_config)
        print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])


if args.evaluate_perp:
    ppl = evaluator.evaluate_perp(user_model)
    print('Perplexity:', ppl)

