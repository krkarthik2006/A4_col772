# 1. Put the HPC shield back up (My mistake for omitting this!)
export CONDA_OVERRIDE_GLIBC=2.17

# 2. Recreate with conda-forge so the Python binary itself is strictly locked to 2.17
conda deactivate
conda env remove -n qwen_scai -y
conda create -n qwen_scai -c conda-forge python=3.10 -y
conda activate qwen_scai

# 3. Lock datasets to version 2.14.0 (which doesn't require PyArrow 21+)
cat <<EOT > requirements.txt
torch>=2.1.0
transformers>=4.45.0
datasets==2.14.0
vllm>=0.6.0
peft>=0.13.0
trl>=0.10.0
tqdm>=4.66.0
accelerate>=0.34.0
sentencepiece>=0.2.0
bitsandbytes==0.42.0
EOT

# 4. The Clean Install
/home/cse/btech/cs1230298/anaconda3/envs/qwen_scai/bin/python3 -m pip install -r requirements.txt