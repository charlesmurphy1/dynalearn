#!/bin/bash
#SBATCH --account=def-aallard
#SBATCH --time=48:00:00
#SBATCH --job-name=sissis_ba_10000
#SBATCH --output=../data/training/sissis_ba_10000/output.out
#SBATCH --gres=gpu:1
#SBATCH --mem=24G

module load python/3.6 scipy-stack mpi4py
source ~/.dynalearn-env/bin/activate
python ss_script.py --config sissis_ba --num_samples 10000 --path_to_data ../data/training/sissis_ba_10000 --path_to_model ../data/models --path_to_summary ../data/training/summary --test 0 --verbose 1
deactivate
