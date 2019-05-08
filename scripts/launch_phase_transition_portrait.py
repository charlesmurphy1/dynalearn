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
import torch
import utilities as u
import tqdm
import matplotlib.pyplot as plt

color_palette = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "purple": "#9A80B9",
    "red": "#d73027",
    "grey": "#525252"
}

def generate_from_model(g, model, init_state, T, num_sample, session):
    N = g.number_of_nodes()
    states = np.zeros((num_sample + 1, T, N))
    adj = nx.to_numpy_array(g)

    p_bar = tqdm.tqdm(range(num_sample * T), 'Model sample')
    for i in range(1, num_sample + 1):
        states[i, 0, :] = init_state * 1
        for t in range(1, T):
            p = model.model.predict([states[i, t - 1, :], adj], steps=1)
            dist = torch.distributions.Categorical(probs=torch.Tensor(p))
            states[i, t, :] = dist.sample()
            p_bar.update()
    p_bar.close()

    return states

def generate_from_dynamics(g, dynamics, init_state, T, num_sample, session):
    N = g.number_of_nodes()
    states = np.zeros((num_sample + 1, T, N))
    dynamics.graph = g

    p_bar = tqdm.tqdm(range(num_sample * T), 'Dynamics sample')
    for i in range(1, num_sample + 1):
        dynamics.states = init_state * 1
        for t in range(T):
            states[i, t, :] = dynamics.states * 1
            dynamics.update()

            p_bar.update()
    p_bar.close()

    return states


def main():
    prs = ap.ArgumentParser(description="Get local transition probability \
                                         figure from path to parameters.")
    prs.add_argument('--path_to_param', '-p', type=str, required=True,
                     help='Path to parameters.')
    prs.add_argument('--num_sample', '-n', default=10,
                     help='Number of samples to use.')
    if len(sys.argv) == 1:
        prs.print_help()
        sys.exit(1)
    args = prs.parse_args()
    with open(args.path_to_param, 'r') as f:
        params = json.load(f)

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    sess = tf.Session()
    num_sample = int(args.num_sample)
    experiment = u.get_experiment(params, False)

    h5file = h5py.File(os.path.join(params["path"], params["name"] + "_model.h5"), 'r')
    experiment.load_hdf5_model(h5file)
    h5file.close()

    h5file = h5py.File(os.path.join(params["path"], params["name"] + "_data.h5"), 'r')
    experiment.load_hdf5_data(h5file)
    h5file.close()

    model = experiment.model
    dynamics = experiment.data_generator.dynamics_model
    N = model.num_nodes

    init_state = np.zeros(N)
    index = np.random.choice(range(N), 5)
    init_state[index] = 1

    g = nx.gnp_random_graph(N, 0.04)


    model_states = generate_from_model(g, model, init_state, 500, num_sample, sess)

    plt.plot(np.mean(np.sum(model_states==0,axis=-1), axis=0), color=color_palette["blue"], marker='o', linestyle='None')
    plt.plot(np.mean(np.sum(model_states==1,axis=-1), axis=0), color=color_palette["orange"], marker='o', linestyle='None')
    plt.plot(np.mean(np.sum(model_states==2,axis=-1), axis=0), color=color_palette["purple"], marker='o', linestyle='None')

    dynamics_states = generate_from_dynamics(g, dynamics, init_state, 500, num_sample, sess)

    plt.plot(np.mean(np.sum(dynamics_states==0,axis=-1), axis=0), color=color_palette["blue"], marker='None', linestyle='-')
    plt.plot(np.mean(np.sum(dynamics_states==1,axis=-1), axis=0), color=color_palette["orange"], marker='None', linestyle='-')
    plt.plot(np.mean(np.sum(dynamics_states==2,axis=-1), axis=0), color=color_palette["purple"], marker='None', linestyle='-')

    plt.show()



if __name__ == '__main__':
    main()