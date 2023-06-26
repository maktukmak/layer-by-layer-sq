# Layer by Layer Quantization

This repo includes scripts to perform static layer by layer quantization instead of full model quantization in order to reduce memory footprint for large models and also enable distrubited quantization. Currently, supports both conventional and smoothquant static per-tensor affine quantization methods.


## CPU
### Setup conda environment
```
conda create -y --name ipex python==3.9
conda activate ipex
conda install astunparse numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses
conda install mkl mkl-include

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu --force-reinstall

python -m pip install intel_extension_for_pytorch -f https://developer.intel.com/ipex-whl-stable-cpu
pip install transformers
```

### Quantization
```
model_name = bigscience/bloom-560m # Supports all bigscience/bloom and facebook/opt models

# Perform conventional int8 quantization (per tensor affine for weights and activations)
python run_layer_quantizer.py --method int8 --model $model_name 

# Perform int8 smoothquant (per tensor affine for weights and activations)
python run_layer_quantizer.py --method sint8 --model $model_name 
```


### Inference
```
# Perform conventional int8 inference
python run_layer_quant_inference.py --method int8 --model $model_name 

# Perform int8 smoothquant inference
python run_layer_quant_inference.py --method sint8 --model $model_name 
```