#!/bin/bash
# ---------------------------------------------------------------------
# Dynalearn submission script.
# ---------------------------------------------------------------------
#SBATCH --job-name=dynalearn_script
#SBATCH --account=def-aallard
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=4000M
#SBATCH --output=scratch/dynalearn/log_files/%x-%j.txt
# ---------------------------------------------------------------------
echo ""
echo "Current working directory: \`pwd\`"
echo "Starting run at: \`date\`"
echo "Submitted by launch_dynamics_learning.sh"
# ---------------------------------------------------------------------
echo ""
echo "Job Name: \$SLURM_JOB_NAME"
echo "Job Array ID / Job ID: \$SLURM_ARRAY_JOB_ID / \$SLURM_JOB_ID"
echo "This is job \$SLURM_ARRAY_TASK_ID out of \$SLURM_ARRAY_TASK_COUNT jobs."
echo ""
# ---------------------------------------------------------------------

calc(){ awk "BEGIN { print "$*" }"; }

making_dir_if_exist () {
    if [[ ! -e ${1} ]]; then
        echo "Making file ${1}."
        mkdir -p ${1}
    fi
}

dynamics="sir"
network="ba"
num_nodes=1000
# density=0.05
density=2
num_sample=10000

# Prepare simulation
PATH_TO_EXP="$HOME/scratch/dynalearn/data"
PATH_TO_SCRIPT="$HOME/packages/dynalearn/scripts"
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
cp "$PATH_TO_SCRIPT/parameters_template.json" "${PATH_TO_EXP}/${FILENAME}/parameters.json"

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

python ${PATH_TO_SCRIPT}/launch_training.py -p ${PATH_TO_EXP}/${FILENAME}/parameters.json

# ---------------------------------------------------------------------
echo "Job finished with exit code \$? at: \`date\`"
echo ""
# ---------------------------------------------------------------------
