#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 00:20:00
#SBATCH --gpus=v100-32:2
#SBATCH --output=/ocean/projects/cis230075p/gzhu/a.out

CUDA_VISIBLE_DEVICES="0,1" accelerate launch --config_file my_config.yaml llama_ft.py \
    --model_name meta-llama/Llama-2-13b-hf \
    --dataset_name timdettmers/openassistant-guanaco \
    --use_peft \
    --load_in_4bit \
    --batch_size 1 \
    --gradient_accumulation_steps 2 \
    --log_with wandb \
    --use_fp16
    # --peft_lora_r 8 \
    # --peft_lora_alpha 32