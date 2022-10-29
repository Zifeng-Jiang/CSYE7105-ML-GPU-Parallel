#!/bin/bash
#SBATCH --job-name=final
#SBATCH --output=final.out
#SBATCH --error=final.err
#SBATCH -p reservation --reservation=csye7105
#SBATCH -N 1
#SBATCH
#SBATCH --gres=gpu:4
module load anaconda3/2022.01

python Final_Project_Zifeng_Jiang_Discovery.py