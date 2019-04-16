#!/bin/bash

# Antoine Allard
# antoineallard.info
# Janvier 2019

# Ce code soumet une tache sur cedar.

# Compilation: g++ -O3 src/solve_analytical_extensive_component_size.cpp -o bin/solve_analytical_extensive_component_size

# =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=
# Ecriture du script de soumission et des commandes a executer.
cat <<END_OF_SCRIPT > script1.pbs
#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for job submission on a Compute Canada cluster.
# ---------------------------------------------------------------------
#SBATCH --job-name=NAME1_extensive_component_size
#SBATCH --account=def-aallard
#SBATCH --time=TIME
#SBATCH --mem-per-cpu=MEM
#SBATCH --output=log_files/analytical_extensive_component_size/NAME1/%x-%j.txt
#SBATCH --array=1-NBTASKS
# ---------------------------------------------------------------------
echo ""
echo "Current working directory: \`pwd\`"
echo "Starting run at: \`date\`"
echo "Submitted by launch_solve_analytical_extensive_component_size.sh"
# ---------------------------------------------------------------------
echo ""
echo "Job Name: \$SLURM_JOB_NAME"
echo "Job Array ID / Job ID: \$SLURM_ARRAY_JOB_ID / \$SLURM_JOB_ID"
echo "This is job \$SLURM_ARRAY_TASK_ID out of \$SLURM_ARRAY_TASK_COUNT jobs."
echo ""
# ---------------------------------------------------------------------

# Filename
fname=NAME1_extensive_component_size_\${SLURM_ARRAY_TASK_ID}.dat

# Scratch directory
scratch_dir=/home/aallard/scratch

# Runs the simulation only if the partial file does not exists.
if [[ ! -f results/analytical/NAME1_extensive_component_size_partial/\${fname} ]]; then

    # Launches the calculation.
    ./bin/solve_analytical_extensive_component_size edgelists/NAME1.edge \${scratch_dir}/\${fname} DELTA_T NBTASKS \$SLURM_ARRAY_TASK_ID

    # Moves the final file to a more permanent directory.
    mv \${scratch_dir}/\${fname} results/analytical/NAME1_extensive_component_size_partial/

else

    echo "The partial file has already be calculated. No further action required. Exiting..."

fi

# ---------------------------------------------------------------------
echo "Job finished with exit code \$? at: \`date\`"
echo ""
# ---------------------------------------------------------------------

exit

END_OF_SCRIPT

# Gets the name of the graph.
name1=$1

# Sets the parameters specific to the graph.
if [[ $name1 == "CayleyTree_z03_l07" ]]; then
    nbtasks=100
    delta_t=0.0001
    # numzero=0.00001
    time="12:00:00"
    mem="256M"
elif [[ $name1 == "CayleyTree_z03_l07_K6_pure_leaf" ]]; then
    nbtasks=100
    delta_t=0.0001
    # numzero=0.00001
    time="48:00:00"
    mem="256M"
elif [[ $name1 == "CayleyTree_z03_l12" ]]; then
    nbtasks=100
    delta_t=0.0001
    # numzero=0.00001
    time="24:00:00"
    mem="256M"
elif [[ $name1 == "CayleyTree_z03_l12_K6_pure_leaf" ]]; then
    nbtasks=100
    delta_t=0.0001
    # numzero=0.00001
    time="48:00:00"
    mem="256M"
elif [[ $name1 == "ExpoGraph_25k" ]]; then
    nbtasks=100
    delta_t=0.0001
    # numzero=0.00001
    time="48:00:00"
    mem="256M"
elif [[ $name1 == "InternetOregon" ]]; then
    nbtasks=100
    delta_t=0.0001
    # numzero=0.00001
    time="48:00:00"
    mem="256M"
else
    echo "Error: unknown graph name. Exiting...."
    exit
fi


# Launches the calculations only they haven't been done already.
if [[ ! -f results/analytical/${name1}_extensive_component_size.dat ]]; then

    # Creates an empty file to mark that the simulations are running (avoid launching twice the same calculation).
    touch results/analytical/${name1}_extensive_component_size.dat

    # Creates a sub-directory to store the log files.
    if [[ ! -d log_files/analytical_extensive_component_size ]]; then
        mkdir log_files/analytical_extensive_component_size
    fi
    if [[ ! -d log_files/analytical_extensive_component_size/$name1 ]]; then
        mkdir log_files/analytical_extensive_component_size/$name1
    fi

    # Creates a sub-directory to store the partial output filenames.
    if [[ ! -d results/analytical/${name1}_extensive_component_size_partial ]]; then
        mkdir results/analytical/${name1}_extensive_component_size_partial
    fi

    #  Copies the model script (see above) file and changes the variables.
    echo $name1
    cp script1.pbs                      script1_${name1}_extensive_component_size.pbs
    sed -i 's,NAME1,'"${name1}"',g'     script1_${name1}_extensive_component_size.pbs
    sed -i 's,NBTASKS,'"${nbtasks}"',g' script1_${name1}_extensive_component_size.pbs
    sed -i 's,DELTA_T,'"${delta_t}"',g' script1_${name1}_extensive_component_size.pbs
    # sed -i 's,NUMZERO,'"${numzero}"',g' script1_${name1}_extensive_component_size.pbs
    sed -i 's,TIME,'"${time}"',g'       script1_${name1}_extensive_component_size.pbs
    sed -i 's,MEM,'"${mem}"',g'         script1_${name1}_extensive_component_size.pbs

    #  Submits the task.
    sbatch script1_${name1}_extensive_component_size.pbs

    # Removes the task-specific model script file.
    rm script1_${name1}_extensive_component_size.pbs

fi

# Deletes the script.
rm script1.pbs

