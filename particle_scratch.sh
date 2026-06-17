#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mem=400gb
#SBATCH --job-name=scratch
#SBATCH --gres=gpu:1
#SBATCH --partition=dev_accelerated
#SBATCH --time=00:30:00

source $(ws_find einspace_ws)/einspace_venv/bin/activate
python einspace/main.py --config configs/particle/re_scratch_particle.yaml --device cuda:0