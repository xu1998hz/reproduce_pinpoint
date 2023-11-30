#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 1:30:00
#SBATCH --gpus=v100-32:2
#SBATCH --output=/ocean/projects/cis230075p/gzhu/reproduce_pinpoint/slurm_out/en-de_23.out

./main_run.sh -i mt_test_data/en-de_wmt_test_wmt23.tsv -b 1 -m 13 -p fp16 -o out/ -t mt -d False