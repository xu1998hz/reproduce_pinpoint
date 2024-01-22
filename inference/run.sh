#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --account=guangleizhu
#SBATCH --partition=taurus
#SBATCH --output=/home/guangleizhu/reproduce_pinpoint/slurm_out/mt_eval_test.out
#SBATCH --error=/home/guangleizhu/reproduce_pinpoint/slurm_out/mt_eval_test_error.out

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

cd /home/guangleizhu/reproduce_pinpoint/inference

# deepspeed --include localhost:0,1,4,5,7 finetune_llama.py
python3 test.py