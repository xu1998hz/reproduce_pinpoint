#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=512GB
#SBATCH --gpus=4
#SBATCH --time=4:00:00
#SBATCH --account=guangleizhu
#SBATCH --partition=aries
#SBATCH --output=/home/guangleizhu/reproduce_pinpoint/finetune/ft_llama_test_4.out
#SBATCH --error=/home/guangleizhu/reproduce_pinpoint/finetune/ft_llama_test_4_error.out

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

cd /home/guangleizhu/reproduce_pinpoint/finetune

deepspeed --num_gpus 4 finetune_llama.py