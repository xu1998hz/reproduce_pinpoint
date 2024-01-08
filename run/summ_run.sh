#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 4:30:00
#SBATCH --gpus=v100-32:2
#SBATCH --output=/ocean/projects/cis230075p/gzhu/reproduce_pinpoint/slurm_out/summ.out

./main_run.sh -i data/summ_test.tsv -b 1 -m 13 -p fp16 -o out/ -t summ -d False