#!/bin/bash
# Minimal SLURM submit wrapper: pass a config as the first argument, e.g.
#   sbatch submit.sh configs/mnist_supcon.yaml
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/output-%j.out
#SBATCH --error=slurm_logs/error-%j.err

python cli.py fit --config "$1"
