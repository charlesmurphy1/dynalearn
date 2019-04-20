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
    prs.add_argument('--path_to_param', '-p', type=str, required=True,
                     help='Path to parameters.')
    prs.add_argument('--save', '-s', type=str, required=False,
                     help='Path where to save.')
    prs.add_argument('--num_sample', '-n', default=1000,
                     help='Number of samples to use.')
    if len(sys.argv) == 1:
        prs.print_help()
        sys.exit(1)
    args = prs.parse_args()
    with open(args.path_to_param, 'r') as f:
        params = json.load(f)


    num_sample = int(args.num_sample)
    experiment = u.get_experiment(params)

    h5file = h5py.File(os.path.join(params["path"], "experiment.h5"), 'r')

    experiment.load_hdf5_model(h5file)
    experiment.load_hdf5_data(h5file)

    graphs = experiment.data_generator.graph_inputs
    states = {}
    targets = {}
    for g in graphs:
        states[g] = experiment.data_generator.state_inputs[g][:num_sample]
        targets[g] = experiment.data_generator.targets[g][:num_sample]

    model = experiment.model
    dynamics = experiment.data_generator.dynamics_model

    N = model.num_nodes
    num_graphs = len(graphs)
    state_label = dynamics.state_label


    # Getting transition probability
    num_states = np.sum([states[g].shape[0] for g in graphs])
    p_bar = tqdm.tqdm(range(num_states), "Transition probabilities")
    dynamics_tp = {}
    model_tp = {}
    for g in graphs:
        dynamics.graph = nx.from_numpy_array(graphs[g])

        dynamics_tp[g] = np.zeros((states[g].shape[0],
                                   states[g].shape[1],
                                   dynamics.num_states))
        model_tp[g] = np.zeros((states[g].shape[0],
                                states[g].shape[1],
                                dynamics.num_states))
        for i, s in enumerate(states[g]):
            dynamics_tp[g][i, :, :] = dynamics.predict(s)
            model_tp[g][i, :, :] = model.model.predict([s, graphs[g]], steps=1)
            p_bar.update()
    p_bar.close()

    # Getting max/min infected degree
    transitions = params["dynamics"]["params"]["relevant_transitions"]
    for g in graphs:
        deg = np.sum(graphs[g], 0)
        kmin = np.min(deg)
        kmax = np.max(deg)

    # degrees = np.arange(kmin, kmax + 1, 1).astype('int')
    degrees = np.arange(N).astype('int')

    # Getting local transition probability
    p_bar = tqdm.tqdm(range(len(graphs)), "Local transition probabilities")
    avg_dynamics_ltp = {(t[0], t[1]): np.zeros(len(degrees)) for t in transitions}

    avg_model_ltp = avg_dynamics_ltp.copy()
    avg_estimate_ltp = avg_dynamics_ltp.copy()
    var_model_ltp = avg_dynamics_ltp.copy()
    var_estimate_ltp = avg_dynamics_ltp.copy()


    for g in graphs:
        dynamics.graph = nx.from_numpy_array(graphs[g])
        in_states = states[g]
        out_states = targets[g]
        dynamics_ltp = dynamics.ltp(in_states)
        model_ltp = dynamics.model_ltp(model.model, in_states)
        estimate_ltp = dynamics.estimate_ltp(in_states, out_states) 
        for t in transitions:
            in_s, out_s = t[0], t[1]
            avg_dynamics_ltp[(in_s, out_s)] = dynamics_ltp[(in_s, out_s)][degrees]
            avg_model_ltp[(in_s, out_s)] = model_ltp[0][(in_s, out_s)][degrees]
            avg_estimate_ltp[(in_s, out_s)] = estimate_ltp[0][(in_s, out_s)][degrees]

            var_model_ltp[(in_s, out_s)] = model_ltp[1][(in_s, out_s)][degrees]
            var_estimate_ltp[(in_s, out_s)] = estimate_ltp[1][(in_s, out_s)][degrees]
        p_bar.update()
    p_bar.close()



    # # Getting attention coefficients
    # p_bar = tqdm.tqdm(range(num_states), "Attention coefficients")
    # attn_layers = u.get_all_attn_layers(model)
    # cond_attn_coeff = {l:{(in_s, out_s): [] for in_s in dynamics.state_label
    #                                      for out_s in dynamics.state_label}
    #                     for l in range(len(attn_layers))}
    # attn_coeff
    # for g in graphs:
    #     adj = graphs[g]
    #     for i, s in enumerate(states[g]):

    #         attn_coeff = []

    #         for j, layer in enumerate(attn_layers):
    #             attn_coeff.append(layer.predict([s, adj], steps=1))

    #         s = s.reshape(adj.shape[0], 1)
    #         for t in transitions:
    #             in_s, out_s = t[0], t[1]
    #             in_l, out_l = state_label[in_s], state_label[out_s]
    #             avail_s = (s==in_l) * (s==out_l).T * adj
    #             for layer in range(len(attn_layers)):
    #                 a = attn_coeff[layer][avail_s==1]
    #                 cond_attn_coeff[layer][(in_s, out_s)].extend(a)
    #         p_bar.update()
    # p_bar.close()

    h5file.close()

    # Writting analytics to file
    if args.save is None:
        h5file = h5py.File(os.path.join(params["path"], "analytics.h5"), 'w')
    else:
        h5file = h5py.File(os.path.join(params["path"], args.save), 'w')


    for g in graphs:
        # Saving transition probabilities
        h5file.create_dataset("/analytics/trans_prob/" + g + "/ground_truth", data=dynamics_tp[g])
        h5file.create_dataset("/analytics/trans_prob/" + g + "/model", data=model_tp[g])

    for t in transitions:
        in_s, out_s = t[0], t[1]

        # Saving local transition probabilities
        group = "/analytics/local_trans_prob/ground_truth/" + in_s + "_to_" + out_s
        val = avg_dynamics_ltp[(in_s, out_s)]
        h5file.create_dataset(group, data=val)

        group = "/analytics/local_trans_prob/model/" + in_s + "_to_" + out_s
        avg, var = avg_model_ltp[(in_s, out_s)], var_model_ltp[(in_s, out_s)]
        h5file.create_dataset(group, data=np.vstack((avg, var)).T)

        group = "/analytics/local_trans_prob/estimate/" + in_s + "_to_" + out_s
        avg, var = avg_estimate_ltp[(in_s, out_s)], var_estimate_ltp[(in_s, out_s)]
        h5file.create_dataset(group, data=np.vstack((avg, var)).T)

        # # Saving attention coefficients
        # for layer in range(len(attn_layers)):
        #     group = "/analytics/attention_coeff_" + str(layer) + "/" + in_s + "_to_" + out_s
        #     h5file.create_dataset(group, data=cond_attn_coeff[layer][(in_s, out_s)])
    h5file.close()

if __name__ == '__main__':
    main()
