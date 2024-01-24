#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 8:00:00
#SBATCH --cpus-per-task=5
#SBATCH --gpus=v100-32:1
#SBATCH --output=/ocean/projects/cis230075p/gzhu/reproduce_pinpoint/slurm_out/mt_eval_test_seg_4.out
#SBATCH --error=/ocean/projects/cis230075p/gzhu/reproduce_pinpoint/slurm_out/mt_eval_test_seg_4_error.out

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

cd /ocean/projects/cis230075p/gzhu/reproduce_pinpoint/inference

# deepspeed --include localhost:0,1,4,5,7 finetune_llama.py
python3 mt_metric.py --seg 4