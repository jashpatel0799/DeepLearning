#!/bin/bash
#SBATCH --job-name=dlops_lab_7
#SBATCH --partition=gpu2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu


module load anaconda/3
pip install -q torchmetrics

python Train_resnet50.py