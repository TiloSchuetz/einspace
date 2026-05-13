#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mem=700gb
#SBATCH --job-name=re_addnist
#SBATCH --gres=gpu:1
#SBATCH --partition=accelerated-h100
#SBATCH --time=36:00:00

source $(ws_find einspace_ws)/einspace_venv/bin/activate
python einspace/main.py --config configs/addnist/re_scratch_addnist.yaml --device cuda:0