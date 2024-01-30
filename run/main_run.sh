#!/bin/bash

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

cd /ocean/projects/cis230075p/gzhu/reproduce_pinpoint

while getopts i:b:m:p:o:t:d:s: flag
do
    case "${flag}" in
        i) data_file=${OPTARG};;
        t) task=${OPTARG};;
        b) batch_size=${OPTARG};;
        m) model_base=${OPTARG};;
        s) model_size=${OPTARG};;
        p) precision=${OPTARG};;
        o) out_dir=${OPTARG};;
        d) debug=${OPTARG};;
        # l) lang=${OPTARG};;
    esac
done

fname=$(basename "$data_file" | sed 's/\.[^.]*$//')
lang=$(echo "$fname" | cut -d'_' -f1)
mkdir -p $out_dir/data
out_data_path="$out_dir/data/${fname}_out.pkl"
comet_model="Unbabel/wmt22-comet-da"

echo "Data File: $data_file"
echo "Task: $task"
echo "Language: $lang"
echo "Batch Size: $batch_size"
echo "Model: ${model_base}-${model_size}b"
echo "Precision: $precision"
echo "Output Directory: $out_dir"
echo "Comet Model: $comet_model"
echo "Debug Mode: $debug"

# echo commands to stdout
# set -x

#run scripts
echo "Running ${model_base}-${model_size}b inference!"
python3 inference/inference.py --model_base "$model_base" --model_size "$model_size" --data_file "$data_file" --out_path "$out_data_path" --precision "$precision" \
     --batch_size "$batch_size" --language "$lang" --task "$task" --debug "$debug"
echo "Saved inference output to $out_data_path"
out_clean_path="$out_dir/data/${fname}_out_clean.pkl"
echo "Cleaning up!"
python3 inference/postprocess.py --data_file "$out_data_path" --out_path "$out_clean_path" --model_base "$model_base" --language "$lang" --task "$task" --debug "$debug"
echo "Saved cleaned output to $out_clean_path"

if [ "$task" = "mt" ]; then
    comet_out="$out_dir/comet_scores_${fname}_${model_base}.json"
    conda activate comet
    echo "Running COMET eval for ${task}!"
    python3 inference/comet_eval.py --data "$out_clean_path" --out_path "$comet_out" --model "$comet_model" 
    echo "Saved COMET eval to $comet_out"
elif [ "$task" = "qa" ]; then
    rouge_out="$out_dir/rouge_${fname}.txt"
    echo "Running ROUGE eval for ${task}!"
    python3 inference/rouge_eval.py --data "$out_clean_path" --out_path "$rouge_out" --task "$task"
    echo "Saved ROUGE eval to $rouge_out"
elif [ "$task" = "summ" ]; then
    rouge_out="$out_dir/rouge_${fname}.txt"
    echo "Running ROUGE eval for ${task}!"
    python3 inference/rouge_eval.py --data "$out_clean_path" --out_path "$rouge_out" --task "$task"
    echo "Saved ROUGE eval to $rouge_out"
else
    echo "Invalid task!"
fi
