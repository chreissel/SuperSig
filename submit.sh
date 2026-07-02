#!/bin/bash
# SLURM submit wrapper for the Harvard Cannon / iaifi_lab cluster, matching
# phlab-neurips25.  Pass a config as the first argument, e.g.
#   sbatch submit.sh configs/jetclass_supcon.yaml
#SBATCH --partition=iaifi_gpu_priority
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/output-%j.out
#SBATCH --error=slurm_logs/error-%j.err

source ~/.bash_profile
mamba activate torch_gpu            # adjust to your environment name if different
cd "$(dirname "$(readlink -f "$0")")"   # run from the repo root (portable)
python cli.py fit --config "$1"
