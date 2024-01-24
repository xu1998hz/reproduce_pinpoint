#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 16:00:00
#SBATCH --cpus-per-task=20
#SBATCH --gpus=v100-32:4
#SBATCH --output=/ocean/projects/cis230075p/gzhu/reproduce_pinpoint/slurm_out/ft_llama.out
#SBATCH --error=/ocean/projects/cis230075p/gzhu/reproduce_pinpoint/slurm_out/ft_llama_error.out

source ~/.bashrc
# module purge
eval "$(conda shell.bash hook)"
conda activate torch2.1

# export PATH=:/home/guangleizhu/.vscode-server/bin/0ee08df0cf4527e40edc9aa28f4b5bd38bbff2b2/bin/remote-cli:/home/guangleizhu/miniconda3/envs/ds/bin:/home/guangleizhu/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/guangleizhu/.dotnet/tools

nvidia-smi

# move to working directory
# this job assumes:
# - all input data is stored in this directory
# - all output should be stored in this directory
# - please note that groupname should be replaced by your groupname
# - username should be replaced by your username
# - path-to-directory should be replaced by the path to your directory where the executable is

# cd /home/guangleizhu/reproduce_pinpoint/finetune
cd /ocean/projects/cis230075p/gzhu/reproduce_pinpoint/finetune

# deepspeed --include localhost:0,1,4,5,7 finetune_llama.py
deepspeed --num_gpus 4 finetune_llama.py