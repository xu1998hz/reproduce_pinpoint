source ~/.bashrc
# module purge
eval "$(conda shell.bash hook)"
conda activate torch2.1

nvidia-smi

# move to working directory
# this job assumes:
# - all input data is stored in this directory
# - all output should be stored in this directory
# - please note that groupname should be replaced by your groupname
# - username should be replaced by your username
# - path-to-directory should be replaced by the path to your directory where the executable is

cd ../finetune

deepspeed --num_gpus 4 finetune_llama.py --lang zh-en
deepspeed --num_gpus 4 finetune_llama.py --lang en-de