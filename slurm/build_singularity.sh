#!/bin/bash

#SBATCH -A evivoli
#SBATCH -p a100
#SBATCH --qos a100_cpu
#SBATCH --time 02:00:00
#SBATCH --gres=gpu:0
#SBATCH --job-name=build_sing
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emanuele.vivoli@unifi.it

singularity build --sandbox --fakeroot ${SCRATCH_A100}/images/<NAME>.sif Singularity