#!/bin/bash

#SBATCH -A evivoli
#SBATCH -p a100
#SBATCH --qos a100_cpu
#SBATCH --time 02:00:00
#SBATCH --gres=gpu:0
#SBATCH --job-name=build_sing
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emanuele.vivoli@unifi.it

singularity exec --nv \
    --env-file $(pwd)/.env \
    --writable -B ${SCRATCH_A100}/logs_cvs:/output \
    --writable -B ${SCRATCH_A100}/data:/data \
    --writable -B /scratch/a100/DATASETS:/dataset \
    --writable -B $(pwd)/:/code ${SCRATCH_A100}/images/torch_21-11.sif \
    python /code/src/cvs/main.py --config_path /code/configs/NeurIPS/clip_contr-10.yaml