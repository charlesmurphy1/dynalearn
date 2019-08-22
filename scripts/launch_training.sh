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
echo "Submitted by launch_dynamics.sh"
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

dynamics="__DYNAMICS__"
network="__NETWORK__"
num_nodes=1000
num_sample=10000

# Prepare simulation
if [ "$HOSTNAME" == "Hector" ] || [ "$HOSTNAME" == "Bernard-Jr" ];
then
    PATH_TO_EXP="$HOME/Documents/ulaval/doctorat/projects/dynalearn/data/training"
    making_dir_if_exist ${PATH_TO_EXP}
    PATH_TO_SCRIPT="$HOME/Documents/ulaval/doctorat/projects/dynalearn/scripts"
else
    source ~/pyenv/bin/activate
    PATH_TO_EXP="$HOME/scratch/dynalearn/data"
    PATH_TO_SCRIPT="$HOME/packages/dynalearn/scripts"
fi
FILENAME="n${num_nodes}_d${num_sample}"

if [[ ${dynamics} = "sir" ]]; then
    PATH_TO_EXP="${PATH_TO_EXP}/sir"
    making_dir_if_exist ${PATH_TO_EXP}
elif [[ ${dynamics} = "sis" ]]; then
    PATH_TO_EXP="${PATH_TO_EXP}/sis"
    making_dir_if_exist ${PATH_TO_EXP}
elif [[ ${dynamics} = "st-sir" ]]; then
    PATH_TO_EXP="${PATH_TO_EXP}/st-sir"
    making_dir_if_exist ${PATH_TO_EXP}
elif [[ ${dynamics} = "st-sis" ]]; then
    PATH_TO_EXP="${PATH_TO_EXP}/st-sis"
    making_dir_if_exist ${PATH_TO_EXP}
elif [[ ${dynamics} = "sis-sis" ]]; then
    PATH_TO_EXP="${PATH_TO_EXP}/sis-sis"
    making_dir_if_exist ${PATH_TO_EXP}
else
    echo "Wrong dynamics type"
    exit 1
fi

if [[ ${network} = "ser" ]]; then
    PATH_TO_EXP="${PATH_TO_EXP}/sparse-erdos-renyi"
    making_dir_if_exist ${PATH_TO_EXP}
    density=0.04
elif [[ ${network} = "der" ]]; then
    PATH_TO_EXP="${PATH_TO_EXP}/dense-erdos-renyi"
    making_dir_if_exist ${PATH_TO_EXP}
    density=0.004
elif [[ ${network} = "ba" ]]; then
    PATH_TO_EXP="${PATH_TO_EXP}/barabasi-albert"
    making_dir_if_exist ${PATH_TO_EXP}
    density=2
else
    echo "Wrong network type"
    exit 1
fi

making_dir_if_exist "${PATH_TO_EXP}/${FILENAME}"
cp "${PATH_TO_SCRIPT}/__PARAM_PATH__" "${PATH_TO_EXP}/${FILENAME}/parameters.json"

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
    sed -i 's,DYNAMICS,'"SIR"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json
elif [[ ${dynamics} = "sis" ]]; then
    sed -i 's,DYNAMICS,'"SIS"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json
elif [[ ${dynamics} = "st-sir" ]]; then
    sed -i 's,DYNAMICS,'"SoftThresholdSIR"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json
elif [[ ${dynamics} = "st-sis" ]]; then
    sed -i 's,DYNAMICS,'"SoftThresholdSIS"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json
elif [[ ${dynamics} = "sis-sis" ]]; then
    sed -i 's,DYNAMICS,'"CooperativeContagionSIS"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json
else
    echo "Wrong dynamics type"
    exit 1
fi

sed -i 's,NUM_SAMPLE,'"${num_sample}"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json
sed -i 's,PATH_TO_EXP,'"${PATH_TO_EXP}/${FILENAME}"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json
sed -i 's,EXP_NAME,'"${network}_${dynamics}_${FILENAME}"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json
sed -i 's,PATH_TO_BEST_MODEL,'"best_weights"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json
sed -i 's,PATH_TO_LAST_MODEL,'"weights"',g'     ${PATH_TO_EXP}/${FILENAME}/parameters.json

python ${PATH_TO_SCRIPT}/training.py -p ${PATH_TO_EXP}/${FILENAME}/parameters.json

# ---------------------------------------------------------------------
echo "Job finished with exit code \$? at: \`date\`"
echo ""
# ---------------------------------------------------------------------
