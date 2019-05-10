#!/bin/bash

PATH_TO_SCRIPT="/home/charles/Documents/ulaval/doctorat/projects/dynalearn/scripts"
PATH_TO_EXP="/home/charles/Documents/ulaval/doctorat/projects/dynalearn/data/sis/dense-erdos-renyi"

num_nodes=(100)
num_sample=(100000)
p=(0.5)
with_truth=(0)
prefix="der"


# Prepare data
for wt in ${with_truth[@]}; do
    for ((i=0; i<${#num_nodes[@]}; ++i)); do
        for d in ${num_sample[@]}; do
            echo "Making ${PATH_TO_EXP}/n${num_nodes[i]}_d${d}_wt${wt}"
            mkdir -p "${PATH_TO_EXP}/n${num_nodes[i]}_d${d}_wt${wt}"
            cp "${PATH_TO_EXP}/parameters_template.json" "${PATH_TO_EXP}/n${num_nodes[i]}_d${d}_wt${wt}/parameters.json"

            sed -i 's,NUM_NODES,'"${num_nodes[i]}"',g'     ${PATH_TO_EXP}/n${num_nodes[i]}_d${d}_wt${wt}/parameters.json
            sed -i 's,DENSITY,'"${p[i]}"',g'     ${PATH_TO_EXP}/n${num_nodes[i]}_d${d}_wt${wt}/parameters.json
            sed -i 's,NUM_SAMPLE,'"${d}"',g'     ${PATH_TO_EXP}/n${num_nodes[i]}_d${d}_wt${wt}/parameters.json
            sed -i 's,PATH_TO_EXP,'"${PATH_TO_EXP}/n${num_nodes[i]}_d${d}_wt${wt}"',g'     ${PATH_TO_EXP}/n${num_nodes[i]}_d${d}_wt${wt}/parameters.json
            sed -i 's,EXP_NAME,'"${prefix}_n${num_nodes[i]}_d${d}_wt${wt}"',g'     ${PATH_TO_EXP}/n${num_nodes[i]}_d${d}_wt${wt}/parameters.json
            sed -i 's,WITH_TRUTH,'"${wt}"',g'     ${PATH_TO_EXP}/n${num_nodes[i]}_d${d}_wt${wt}/parameters.json

            python ${PATH_TO_SCRIPT}/launch_training_script.py -p ${PATH_TO_EXP}/n${num_nodes[i]}_d${d}_wt${wt}/parameters.json
            python ${PATH_TO_SCRIPT}/launch_analytics_script.py -p ${PATH_TO_EXP}/n${num_nodes[i]}_d${d}_wt${wt}/parameters.json -n 1000
            python ${PATH_TO_SCRIPT}/figure_local_transition_probability.py -p  ${PATH_TO_EXP}/n${num_nodes[i]}_d${d}_wt${wt}/parameters.json -s "ltp.png"
            python ${PATH_TO_SCRIPT}/figure_attention_coeff.py -p  ${PATH_TO_EXP}/n${num_nodes[i]}_d${d}_wt${wt}/parameters.json -s "attn.png"
            python ${PATH_TO_SCRIPT}/figure_generalization.py -p  ${PATH_TO_EXP}/n${num_nodes[i]}_d${d}_wt${wt}/parameters.json -s "gen.png"
        done
    done
done