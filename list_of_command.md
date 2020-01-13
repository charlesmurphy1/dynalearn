module load python/3.6.3
source ~/pyenv/dynalearn-env/bin/activate
pip install numpy --no-index
pip install tensorflow_gpu --no-index
pip install torch_gpu --no-index
pip install networkx
pip install matplotlib
pip install tqdm
pip install abcpy
deactivate

<!-- virtualenv --no-download ~/pyenv/dynalearn-env -->
