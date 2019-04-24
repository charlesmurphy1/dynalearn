#!/bin/bash
#SBATCH --account=murphy9
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=4000M               # memory (per node)
#SBATCH --time=0-03:00            # time (DD-HH:MM)

PATH_TO_EXP="../data/ba10"

python launch_training_script.py -p $PATH_TO_EXP"/parameters.json"
python launch_analytics_script.py -p $PATH_TO_EXP"/parameters.json"
python figure_local_transition_probability.py -p $PATH_TO_EXP"/parameters.json" -s "ltp.png"
python figure_local_transition_probability.py -p $PATH_TO_EXP"/parameters.json" -s "ltp.pdf"
# python figure_markov_matrix.py -p $PATH_TO_EXP+"/parameters.json"

