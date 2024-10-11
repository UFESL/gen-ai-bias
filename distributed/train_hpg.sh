#!/bin/bash
# Adjust any of these parameters if needed. 
# Try not to adjust the ntasks, mem, cpus, or gpus unless you run out of resources to allow other students in our group to also run their experiments.
#SBATCH --job-name=sd
#SBATCH --ntasks=1
#SBATCH --mem=32gb
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=sd_%j.log
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1

# Location to conda environment. torch is the name of my conda environment. You MUST include /bin after the name of your environment.
env_path=/blue/prabhat/e.andrews/.conda/envs/torch/bin
export PATH=$env_path:$PATH

# Command you would normally run to run the Python file.
python test.py