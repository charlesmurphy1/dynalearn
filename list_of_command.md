## To make new virtual environment

module load python/3.6 scipy-stack mpi4py
virtualenv .dynalearn-env/
source .dynalearn-env/bin/activate
<!-- pip install tensorflow_gpu==1.12  -->
pip install abcpy numpy networkx tqdm matplotlib psutil
pip install --no-index torch==1.7.1
pip install --no-index torch_scatter
pip install --no-index torch_sparse
pip install --no-index torch_cluster
pip install --no-index torch_spline_conv
pip install --no-index torch_geometric
python ~/codes/dynalearn/setup.py develop
deactivate

## To launch job
