#!/bin/bash
#SBATCH --account=murphy9
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=4000M               # memory (per node)
#SBATCH --time=0-03:00            # time (DD-HH:MM)


python test_for_cedar.py