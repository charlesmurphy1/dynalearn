import dynalearn as dl
import h5py
import numpy as np
import os 
import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.backend as K
import json
import utilities as u

with open('parameters.json', 'r') as f:
    params = json.load(f)

experiment = u.get_experiment(params)

for i in range(params["data_generator"]["params"]["num_graphs"]):
    experiment.generate_data(params["data_generator"]["params"]["num_sample"],
                             params["data_generator"]["params"]["T"],
                             gamma=params["data_generator"]["params"]["gamma"])

experiment.train_model(params["training"]["epochs"],
                       params["training"]["steps_per_epoch"],
                       verbose=1)

h5file = h5py.File(os.path.join(params["path"],
                                params["experiment_name"] + ".h5"),
                   'w')
experiment.save_hdf5_all(h5file)
h5file.close()
