#!/bin/bash
#PBS -A iwg-952-aa
#PBS -l walltime=12:00:00
#PBS -l nodes=1:gpus=1
#PBS -r n
#PBS -o /home/murphy9/exif-submits/resnet18.out
#PBS -e /home/murphy9/exif-submits/resnet18.err

module load cuda/8.0.44

source $HOME/pyenv/bin/activate

python script.py