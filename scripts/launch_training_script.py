import argparse as ap
import dynalearn as dl
import h5py
import json
import networkx as nx
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K
import utilities as u
import tqdm




def main():
    prs = ap.ArgumentParser(description="Get local transition probability \
                                         figure from path to parameters.")
    prs.add_argument('--path', '-p', type=str, required=True,
                     help='Path to parameters.')

    if len(sys.argv) == 1:
        prs.print_help()
        sys.exit(1)
    args = prs.parse_args()

    with open(args.path, 'r') as f:
        print(args.path)
        params = json.load(f)

    print("Building experiment\n-------------------")
    experiment = u.get_experiment(params, True)
    experiment.model.model.summary()

    if experiment.model.num_nodes < 500:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    print("Training:\n--------")
    experiment.train_model(params["training"]["epochs"],
                           params["training"]["steps_per_epoch"],
                           validation_steps=1000,
                           verbose=1)

    h5file = h5py.File(os.path.join(params["path"], params["name"] + "_model.h5"), 'w')
    experiment.save_hdf5_model(h5file)
    h5file.close()

    h5file = h5py.File(os.path.join(params["path"], params["name"] + "_data.h5"), 'w')
    experiment.save_hdf5_data(h5file)
    h5file.close()

    h5file = h5py.File(os.path.join(params["path"], params["name"] + "_optimizer.h5"), 'w')
    experiment.save_hdf5_optimizer(h5file)
    h5file.close()

    h5file = h5py.File(os.path.join(params["path"], params["name"] + "_history.h5"), 'w')
    experiment.save_hdf5_history(h5file)
    h5file.close()



if __name__ == '__main__':
    main()
