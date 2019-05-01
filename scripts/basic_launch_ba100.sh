#!/bin/bash
#SBATCH --account=murphy9
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=4000M               # memory (per node)
#SBATCH --time=0-03:00            # time (DD-HH:MM)

PATH_TO_EXP="../data/ba100/1k"

python launch_training_script.py -p $PATH_TO_EXP"/parameters.json" #> $PATH_TO_EXP"/train.log"
python launch_analytics_script.py -p $PATH_TO_EXP"/parameters.json"  #> $PATH_TO_EXP"/analytics.log"

python figure_local_transition_probability.py -p $PATH_TO_EXP"/parameters.json" -s "ltp.png"
python figure_local_transition_probability.py -p $PATH_TO_EXP"/parameters.json" -s "ltp.pdf"

python figure_star_ltp.py -p $PATH_TO_EXP"/parameters.json" -s "star_ltp.png"
python figure_star_ltp.py -p $PATH_TO_EXP"/parameters.json" -s "star_ltp.pdf"

python figure_markov_matrix.py -p $PATH_TO_EXP"/parameters.json" -s "markov_matrix.png"
python figure_markov_matrix.py -p $PATH_TO_EXP"/parameters.json" -s "markov_matrix.pdf"

python figure_loss_per_degree.py -p $PATH_TO_EXP"/parameters.json" -s "loss.png"
python figure_loss_per_degree.py -p $PATH_TO_EXP"/parameters.json" -s "loss.pdf"

PATH_TO_EXP="../data/ba100/5k"

python launch_training_script.py -p $PATH_TO_EXP"/parameters.json" #> $PATH_TO_EXP"/train.log"
python launch_analytics_script.py -p $PATH_TO_EXP"/parameters.json"  #> $PATH_TO_EXP"/analytics.log"

python figure_local_transition_probability.py -p $PATH_TO_EXP"/parameters.json" -s "ltp.png"
python figure_local_transition_probability.py -p $PATH_TO_EXP"/parameters.json" -s "ltp.pdf"

python figure_star_ltp.py -p $PATH_TO_EXP"/parameters.json" -s "star_ltp.png"
python figure_star_ltp.py -p $PATH_TO_EXP"/parameters.json" -s "star_ltp.pdf"

python figure_markov_matrix.py -p $PATH_TO_EXP"/parameters.json" -s "markov_matrix.png"
python figure_markov_matrix.py -p $PATH_TO_EXP"/parameters.json" -s "markov_matrix.pdf"

python figure_loss_per_degree.py -p $PATH_TO_EXP"/parameters.json" -s "loss.png"
python figure_loss_per_degree.py -p $PATH_TO_EXP"/parameters.json" -s "loss.pdf"

PATH_TO_EXP="../data/ba100/10k"

python launch_training_script.py -p $PATH_TO_EXP"/parameters.json" #> $PATH_TO_EXP"/train.log"
python launch_analytics_script.py -p $PATH_TO_EXP"/parameters.json"  #> $PATH_TO_EXP"/analytics.log"

python figure_local_transition_probability.py -p $PATH_TO_EXP"/parameters.json" -s "ltp.png"
python figure_local_transition_probability.py -p $PATH_TO_EXP"/parameters.json" -s "ltp.pdf"

python figure_star_ltp.py -p $PATH_TO_EXP"/parameters.json" -s "star_ltp.png"
python figure_star_ltp.py -p $PATH_TO_EXP"/parameters.json" -s "star_ltp.pdf"

python figure_markov_matrix.py -p $PATH_TO_EXP"/parameters.json" -s "markov_matrix.png"
python figure_markov_matrix.py -p $PATH_TO_EXP"/parameters.json" -s "markov_matrix.pdf"

python figure_loss_per_degree.py -p $PATH_TO_EXP"/parameters.json" -s "loss.png"
python figure_loss_per_degree.py -p $PATH_TO_EXP"/parameters.json" -s "loss.pdf"

PATH_TO_EXP="../data/ba100/50k"

python launch_training_script.py -p $PATH_TO_EXP"/parameters.json" #> $PATH_TO_EXP"/train.log"
python launch_analytics_script.py -p $PATH_TO_EXP"/parameters.json"  #> $PATH_TO_EXP"/analytics.log"

python figure_local_transition_probability.py -p $PATH_TO_EXP"/parameters.json" -s "ltp.png"
python figure_local_transition_probability.py -p $PATH_TO_EXP"/parameters.json" -s "ltp.pdf"

python figure_star_ltp.py -p $PATH_TO_EXP"/parameters.json" -s "star_ltp.png"
python figure_star_ltp.py -p $PATH_TO_EXP"/parameters.json" -s "star_ltp.pdf"

python figure_markov_matrix.py -p $PATH_TO_EXP"/parameters.json" -s "markov_matrix.png"
python figure_markov_matrix.py -p $PATH_TO_EXP"/parameters.json" -s "markov_matrix.pdf"

python figure_loss_per_degree.py -p $PATH_TO_EXP"/parameters.json" -s "loss.png"
python figure_loss_per_degree.py -p $PATH_TO_EXP"/parameters.json" -s "loss.pdf"

PATH_TO_EXP="../data/ba100/100k"

python launch_training_script.py -p $PATH_TO_EXP"/parameters.json" #> $PATH_TO_EXP"/train.log"
python launch_analytics_script.py -p $PATH_TO_EXP"/parameters.json"  #> $PATH_TO_EXP"/analytics.log"

python figure_local_transition_probability.py -p $PATH_TO_EXP"/parameters.json" -s "ltp.png"
python figure_local_transition_probability.py -p $PATH_TO_EXP"/parameters.json" -s "ltp.pdf"

python figure_star_ltp.py -p $PATH_TO_EXP"/parameters.json" -s "star_ltp.png"
python figure_star_ltp.py -p $PATH_TO_EXP"/parameters.json" -s "star_ltp.pdf"

python figure_markov_matrix.py -p $PATH_TO_EXP"/parameters.json" -s "markov_matrix.png"
python figure_markov_matrix.py -p $PATH_TO_EXP"/parameters.json" -s "markov_matrix.pdf"

python figure_loss_per_degree.py -p $PATH_TO_EXP"/parameters.json" -s "loss.png"
python figure_loss_per_degree.py -p $PATH_TO_EXP"/parameters.json" -s "loss.pdf"
