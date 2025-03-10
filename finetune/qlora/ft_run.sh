#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 3:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gpus=v100-32:2
#SBATCH --output=/ocean/projects/cis230075p/gzhu/reproduce_pinpoint/slurm_out/ft_qlora.out
#SBATCH --error=/ocean/projects/cis230075p/gzhu/reproduce_pinpoint/slurm_out/ft_qlora_error.out

source ~/.bashrc
# module purge
eval "$(conda shell.bash hook)"
conda activate torch2.1

nvidia-smi
cd /ocean/projects/cis230075p/gzhu/reproduce_pinpoint/qlora


CUDA_VISIBLE_DEVICES="0,1" accelerate launch --config_file my_config.yaml llama_ft_qlora.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset_name timdettmers/openassistant-guanaco \
    --use_peft \
    --load_in_4bit \
    --batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-5 \
    --log_with wandb \
    --use_fp16 \
    --seq_length 512 \
    --flash_atten \
    --peft_lora_r 128 \
    --peft_lora_alpha 16 \
    --num_train_epochs 5

# wait for 60 seconds for wandb to upload
sleep 60
# python3 eval.py

# 128 + 32 linear decay loss
#accelerate launch --multi_gpu --num_machines 1  --num_processes 8 my_accelerate_script.py
#torchrun --nnodes 1  --nproc_per_node 8 my_torch_script.py
