#!/bin/bash
##SBATCH --job-name=dialogpt_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G          # Request 16GB (adjust as needed)
#SBATCH --account=def-bengioy
source ~/.bashrc
conda activate llm_finetune_3_10

python quick_finetune_demo.py
