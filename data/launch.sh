#!/bin/bash

calc(){ awk "BEGIN { print "$*" }"; }

making_dir_if_exist () {
    if [[ ! -e ${1} ]]; then
        echo "Making file ${1}."
        mkdir -p ${1}
    fi
}

dynamics="sir"
network="ba"
num_nodes=100
# density=0.05
density=2
num_sample=10000

# Prepare simulation
PATH_TO_EXP="/home/charles/Documents/ulaval/doctorat/projects/dynalearn/data"
PATH_TO_SCRIPT="/home/charles/Documents/ulaval/doctorat/projects/dynalearn/scripts"
FILENAME="n${num_nodes}_d${num_sample}"

if [[ ${dynamics} = "sir" ]]; then
    PATH_TO_EXP="${PATH_TO_EXP}/sir"
    making_dir_if_exist ${PATH_TO_EXP}
elif [[ ${dynamics} = "sis" ]]; then
    PATH_TO_EXP="${PATH_TO_EXP}/sis"
    making_dir_if_exist ${PATH_TO_EXP}

else
    echo "Wrong dynamics type"
    exit 1
fi

if [[ ${network} = "ser" ]]; then
    PATH_TO_EXP="${PATH_TO_EXP}/sparse-erdos-renyi"
    making_dir_if_exist ${PATH_TO_EXP}
elif [[ ${network} = "der" ]]; then
    PATH_TO_EXP="${PATH_TO_EXP}/dense-erdos-renyi"
    making_dir_if_exist ${PATH_TO_EXP}
elif [[ ${network} = "ba" ]]; then
    PATH_TO_EXP="${PATH_TO_EXP}/barabasi-albert"
    making_dir_if_exist ${PATH_TO_EXP}
else
    echo "Wrong network type"
    exit 1
fi

making_dir_if_exist "${PATH_TO_EXP}/${FILENAME}"
cp "./parameters_template.json" "${PATH_TO_EXP}/${FILENAME}/parameters.json"

if [[ ${network} = "ser" ]]; then
    sed -i 's,NETWORK,'"ERGraph"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json
    sed -i 's,NUM_NODES,'"${num_nodes}"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json
    sed -i 's,DENSITYLABEL,'"p"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json
    sed -i 's,DENSITY,'"${density}"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json
elif [[ ${network} = "der" ]]; then
    sed -i 's,NETWORK,'"ERGraph"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json
    sed -i 's,NUM_NODES,'"${num_nodes}"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json
    sed -i 's,DENSITYLABEL,'"p"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json
    sed -i 's,DENSITY,'"${density}"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json
elif [[ ${network} = "ba" ]]; then
    sed -i 's,NETWORK,'"BAGraph"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json
    sed -i 's,NUM_NODES,'"${num_nodes}"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json
    sed -i 's,DENSITYLABEL,'"M"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json
    sed -i 's,DENSITY,'"${density}"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json
else
    echo "Wrong network type"
    exit 1
fi

if [[ ${dynamics} = "sir" ]]; then
    sed -i 's,DYNAMICS,'"SIRDynamics"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json
elif [[ ${dynamics} = "sis" ]]; then
    sed -i 's,DYNAMICS,'"SISDynamics"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json
else
    echo "Wrong dynamics type"
    exit 1
fi

sed -i 's,NUM_SAMPLE,'"${num_sample}"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json
sed -i 's,PATH_TO_EXP,'"${PATH_TO_EXP}/${FILENAME}"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json
sed -i 's,EXP_NAME,'"${network}_${dynamics}_${FILENAME}"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json

with_best="1"

python ${PATH_TO_SCRIPT}/launch_training_script.py -p ${PATH_TO_EXP}/${FILENAME}/parameters.json
# if [[ ${with_best} = "0" ]]; then
#     python ${PATH_TO_SCRIPT}/launch_analytics_script.py -p ${PATH_TO_EXP}/${FILENAME}/parameters.json -b 0 --kmax 100
#     python ${PATH_TO_SCRIPT}/figure_local_transition_probability.py -p  ${PATH_TO_EXP}/${FILENAME}/parameters.json -s "ltp_last.png"
#     python ${PATH_TO_SCRIPT}/figure_attention_coeff.py -p  ${PATH_TO_EXP}/${FILENAME}/parameters.json -s "attn_last.png"
#     python ${PATH_TO_SCRIPT}/figure_generalization.py -p  ${PATH_TO_EXP}/${FILENAME}/parameters.json -s "gen_last.png"
# else
#     python ${PATH_TO_SCRIPT}/launch_analytics_script.py -p ${PATH_TO_EXP}/${FILENAME}/parameters.json -b 1 --kmax 100
#     python ${PATH_TO_SCRIPT}/figure_local_transition_probability.py -p  ${PATH_TO_EXP}/${FILENAME}/parameters.json -s "ltp_best.png"
#     python ${PATH_TO_SCRIPT}/figure_attention_coeff.py -p  ${PATH_TO_EXP}/${FILENAME}/parameters.json -s "attn_best.png"
#     python ${PATH_TO_SCRIPT}/figure_generalization.py -p  ${PATH_TO_EXP}/${FILENAME}/parameters.json -s "gen_best.png"
# fi
