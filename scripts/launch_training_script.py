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
    prs.add_argument('--save', '-s', type=str, required=False,
                     help='Path where to save.')

    if len(sys.argv) == 1:
        prs.print_help()
        sys.exit(1)
    args = prs.parse_args()

    with open(args.path, 'r') as f:
        params = json.load(f)

    print("Building experiment\n-------------------")
    experiment = u.get_experiment(params)

    print("Building dataset\n----------------")
    p_bar = tqdm.tqdm(range(params["data_generator"]["params"]["num_graphs"]*
                            params["data_generator"]["params"]["num_sample"]))
    for i in range(params["data_generator"]["params"]["num_graphs"]):
        experiment.generate_data(params["data_generator"]["params"]["num_sample"],
                                 params["data_generator"]["params"]["T"],
                                 gamma=params["data_generator"]["params"]["gamma"],
                                 progress_bar=p_bar)

    p_bar.close()

    print("Training\n--------")
    experiment.train_model(params["training"]["epochs"],
                           params["training"]["steps_per_epoch"],
                           verbose=1)

    if args.save is None:
        h5file = h5py.File(os.path.join(params["path"], "experiment.h5"), 'w')
    else:
        h5file = h5py.File(os.path.join(params["path"], args.save), 'w')
    experiment.save_hdf5_all(h5file)
    h5file.close()

if __name__ == '__main__':
    main()
