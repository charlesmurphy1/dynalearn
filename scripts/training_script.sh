#!/bin/bash
# ---------------------------------------------------------------------
# Dynalearn submission script.
# ---------------------------------------------------------------------

# Prepare simulation
if [ "$HOSTNAME" == "Hector" ] || [ "$HOSTNAME" == "Bernard-Jr" ];
then
    PATH_TO_SCRIPT="$HOME/Documents/ulaval/doctorat/projects/dynalearn/scripts"
else
    PATH_TO_SCRIPT="$HOME/packages/dynalearn/scripts"
fi

dynamics=("sis-sis")
networks=("ba" "ser")
temp=0

for net in ${networks[*]}; do
    for dyna in ${dynamics[*]}; do
        cp "$PATH_TO_SCRIPT/launch_training.sh" "$PATH_TO_SCRIPT/training_scripts/launch_training_${dyna}_${temp}.sh"
        sed -i 's,__DYNAMICS__,'"${dyna}"',g'     $PATH_TO_SCRIPT/training_scripts/launch_training_${dyna}_${temp}.sh
        sed -i 's,__NETWORK__,'"${net}"',g'     $PATH_TO_SCRIPT/training_scripts/launch_training_${dyna}_${temp}.sh
        sed -i 's,__PARAM_PATH__,'"param_temp_${dyna}.json"',g'     $PATH_TO_SCRIPT/training_scripts/launch_training_${dyna}_${temp}.sh
        bash $PATH_TO_SCRIPT/training_scripts/launch_training_${dyna}_${temp}.sh
        rm $PATH_TO_SCRIPT/training_scripts/launch_training_${dyna}_${temp}.sh
    done
done
