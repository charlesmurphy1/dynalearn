## To make new virtual environment

module load python/3.6 scipy-stack mpi4py
virtualenv .dynalearn-env/
source .dynalearn-env/bin/activate
pip install tensorflow_gpu==1.12 torch abcpy numpy networkx tqdm matplotlib
pip wheel /home/murphy9/projects/def-aallard/murphy9/sources/dynalearn
pip install /home/murphy9/projects/def-aallard/murphy9/sources/dynalearn
deactivate

## To launch job
