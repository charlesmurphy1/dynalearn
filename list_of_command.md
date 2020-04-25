## To make new virtual environment

module load python/3.6 scipy-stack mpi4py
virtualenv .dynalearn-env/
source .dynalearn-env/bin/activate
pip install tensorflow_gpu==1.12 torch abcpy numpy networkx tqdm matplotlib
pip install --no-index torch==1.4.0 torch_cluster==1.4.5 torch_scatter==2.0.3 torch_sparse==0.5.1 torch_spline_conv==1.1.1
pip install torch_geometric
pip wheel /home/murphy9/projects/def-aallard/murphy9/sources/dynalearn
pip install /home/murphy9/projects/def-aallard/murphy9/sources/dynalearn
deactivate

## To launch job
