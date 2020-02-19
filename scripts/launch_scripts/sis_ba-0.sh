#!/bin/bash
#SBATCH --account=def-aallard
#SBATCH --time=24:00:00
#SBATCH --job-name=sis_ba_nn1000
#SBATCH --output=../data/netsize/sis_ba_nn1000/output.out
#SBATCH --gres=gpu:1
#SBATCH --mem=24G

module load python/3.6 scipy-stack mpi4py
source ~/.dynalearn-env/bin/activate
python training_script.py --config sis_ba --num_samples 1000 --num_nodes 1000 --resampling_time 2 --suffix nn1000 --path_to_data ../data/netsize/sis_ba_nn1000 --path_to_model ../data/netsize/models --path_to_summary ../data/netsize/summary --test 1 --verbose 2
deactivate
