#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 4:30:00
#SBATCH --gpus=v100-32:1
#SBATCH --output=/ocean/projects/cis230075p/gzhu/reproduce_pinpoint/slurm_out/qa.out

./main_run.sh -i asqa_test.tsv -b 1 -m 7 -p fp16 -o out/ -t qa -d False