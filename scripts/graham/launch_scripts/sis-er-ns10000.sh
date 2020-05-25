#!/bin/bash
#SBATCH --account=def-aallard
#SBATCH --time=24:00:00
#SBATCH --job-name=sis-er-ns10000
#SBATCH --output=/home/murphy9/projects/def-aallard/murphy9/data/dynalearn-data/semi-exact/outputs/sis-er-ns10000.out
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=24G

module load python/3.6 scipy-stack mpi4py
source /home/murphy9/.dynalearn-env/bin/activate
python /home/murphy9/source/dynalearn/scripts/training_script.py --config sis-er --name sis-er-ns10000 --num_samples 10000 --num_nodes 1000 --resampling_time 2 --batch_size 1 --with_truth 1 --mode complete --path /home/murphy9/projects/def-aallard/murphy9/data/dynalearn-data/semi-exact/full_data --path_to_best /home/murphy9/projects/def-aallard/murphy9/data/dynalearn-data/semi-exact/best --path_to_summary /home/murphy9/projects/def-aallard/murphy9/data/dynalearn-data/semi-exact/summary --seed 0 --verbose 2
deactivate
