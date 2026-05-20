#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mem=400gb
#SBATCH --job-name=re_particle
#SBATCH --gres=gpu:1
#SBATCH --partition=accelerated-h100
#SBATCH --time=10:00:00

source $(ws_find einspace_ws)/einspace_venv/bin/activate
python einspace/main.py --config configs/particle/re_scratch_particle.yaml --device cuda:0