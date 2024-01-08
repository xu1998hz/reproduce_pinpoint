#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 24:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --output=/ocean/projects/cis230075p/gzhu/reproduce_pinpoint/slurm_out/mt.out

./main_run.sh -i mt_test_data/en-de_wmt_test_wmt22.tsv -b 1 -m mistral -s 7 -p fp16 -o out/ -t mt -d False
./main_run.sh -i mt_test_data/en-de_wmt_test_wmt23.tsv -b 1 -m mistral -s 7 -p fp16 -o out/ -t mt -d False
./main_run.sh -i mt_test_data/zh-en_wmt_test_wmt22.tsv -b 1 -m mistral -s 7 -p fp16 -o out/ -t mt -d False
./main_run.sh -i mt_test_data/zh-en_wmt_test_wmt23.tsv -b 1 -m mistral -s 7 -p fp16 -o out/ -t mt -d False