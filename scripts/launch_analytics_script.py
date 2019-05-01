import argparse as ap
import dynalearn as dl
import h5py
import json
import networkx as nx
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import utilities as u
import tqdm


def compute_transition_probability(dynamics, model, states, graphs):
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
    return dynamics_tp, model_tp


def compute_markov_matrix(dynamics, model, states, graphs):
    # Compute Markov matrices and occurence
    N = model.num_nodes
    state_label = dynamics.state_label
    num_graphs = len(graphs)
    num_all_states = len(state_label)**N

    all_states = np.zeros((num_all_states, N)).astype("int")
    dynamics_markovmat = {}
    model_markovmat = {}

    for i in range(1, num_all_states):
        all_states[i] = u.increment_int_from_base(all_states[i - 1],
                                                  len(state_label))

    p_bar = tqdm.tqdm(range(num_all_states * num_graphs), "Markov matrix")
    for g in graphs:
        dynamics_markovmat[g] = np.zeros((num_all_states, num_all_states))
        model_markovmat[g] = np.zeros((num_all_states, num_all_states))

        for i, s in enumerate(all_states):
            dynamics_p = dynamics.predict(s)
            model_p = model.model.predict([s, graphs[g]], steps=1)
            dynamics_logp = np.trace(np.log(dynamics_p.T[all_states]),
                                     axis1=1, axis2=2)
            model_logp = np.trace(np.log(model_p.T[all_states]),
                                  axis1=1, axis2=2)
            dynamics_logp[dynamics_logp<-50] = -50
            model_logp[model_logp<-50] = -50
            dynamics_markovmat[g][:, i] = dynamics_logp
            model_markovmat[g][:, i] = model_logp
            p_bar.update()
    p_bar.close()

    return dynamics_markovmat, model_markovmat


def compute_state_occurence(dynamics, model, states, graphs):
    N = model.num_nodes
    state_label = dynamics.state_label
    num_all_states = len(state_label)**N

    num_states = np.sum([states[g].shape[0] for g in graphs])
    p_bar = tqdm.tqdm(range(num_states), "State occurence")
    occurence = {}
    for g in graphs:
        occurence[g] = np.zeros((num_all_states, num_all_states))
        for i in range(len(states[g]) - 1):
            in_state = states[g][i]
            out_state = states[g][i + 1]
            in_i = u.base_to_int(in_state, len(state_label))
            out_i = u.base_to_int(out_state, len(state_label))
            occurence[g][out_i, in_i] += 1
            p_bar.update()
    p_bar.close()
    return occurence


def compute_ltp(dynamics, model, states, targets, graphs, rel_trans):
    N = model.num_nodes
    degrees = np.arange(N).astype('int')
    avg_dynamics_ltp = {(t[0], t[1]): np.zeros(len(degrees)) for t in rel_trans}

    avg_model_ltp = avg_dynamics_ltp.copy()
    avg_estimate_ltp = avg_dynamics_ltp.copy()
    var_model_ltp = avg_dynamics_ltp.copy()
    var_estimate_ltp = avg_dynamics_ltp.copy()


    p_bar = tqdm.tqdm(range(len(graphs)), "Local transition probabilities")
    for g in graphs:
        dynamics.graph = nx.from_numpy_array(graphs[g])
        in_states = states[g]
        out_states = targets[g]
        dynamics_ltp = dynamics.ltp(in_states)
        model_ltp = dynamics.model_ltp(model.model, in_states)
        estimate_ltp = dynamics.estimate_ltp(in_states, out_states) 
        for t in rel_trans:
            in_s, out_s = t[0], t[1]
            avg_dynamics_ltp[(in_s, out_s)] = dynamics_ltp[(in_s, out_s)][degrees]
            avg_model_ltp[(in_s, out_s)] = model_ltp[0][(in_s, out_s)][degrees]
            avg_estimate_ltp[(in_s, out_s)] = estimate_ltp[0][(in_s, out_s)][degrees]

            var_model_ltp[(in_s, out_s)] = model_ltp[1][(in_s, out_s)][degrees]
            var_estimate_ltp[(in_s, out_s)] = estimate_ltp[1][(in_s, out_s)][degrees]
        p_bar.update()
    p_bar.close()

    return avg_dynamics_ltp, avg_model_ltp, avg_estimate_ltp,\
           var_model_ltp, var_estimate_ltp


def compute_star_ltp(dynamics, model, rel_trans):
    N = model.num_nodes
    degrees = np.arange(N).astype('int')

    star_g = dl.graphs.StarGraph(N)
    name, g = star_g.generate()
    graphs = {name: nx.to_numpy_array(g)}
    _states = np.zeros((len(rel_trans) * N, N))
    for i, t in enumerate(rel_trans):
        in_l = dynamics.state_label[t[0]]
        _states[i * N:(i + 1) * N, 0] = in_l
    for i in range(1, N):
        for j in range(len(rel_trans)):
            _states[i + N * j, 1:i] = 1
    states = {name: _states}

    dynamics_star_ltp = {(t[0], t[1]): np.zeros(len(degrees)) for t in rel_trans}
    model_star_ltp = {(t[0], t[1]): np.zeros(len(degrees)) for t in rel_trans}

    p_bar = tqdm.tqdm(range(len(rel_trans) * N), "Star Local transition probabilities")
    for g in graphs:
        dynamics.graph = nx.from_numpy_array(graphs[g])
        in_states = states[g]

        for s in in_states:
            dynamics_core_p = dynamics.predict(s)[0, :]
            model_core_p = model.model.predict([s, graphs[g]], steps=1)[0, :]
            num_infected = int(np.sum(s[1:] == 1))
            for t in rel_trans:
                in_s, out_s = t[0], t[1]
                if s[0] == dynamics.state_label[in_s]:
                    out_l = dynamics.state_label[out_s]
                    dynamics_star_ltp[(in_s, out_s)][num_infected] = dynamics_core_p[out_l]
                    model_star_ltp[(in_s, out_s)][num_infected] = model_core_p[out_l]
            p_bar.update()
    p_bar.close()

    return dynamics_star_ltp, model_star_ltp


def compute_attn_coeff(dynamics, model, attn_layers, states, graphs):
    N = model.num_nodes
    states = {g: states[g][:20] for g in states}
    num_states = np.sum([states[g].shape[0] for g in graphs])
    state_label = dynamics.state_label

    # Getting attention coefficients
    p_bar = tqdm.tqdm(range(num_states), "Attention coefficients")
    cond_attn = {l:{(in_s, out_s): []
                    for in_s in state_label
                    for out_s in state_label}
                 for l in range(len(attn_layers))}

    attn_coeff = {}
    for g in graphs:
        adj = graphs[g]
        attn_coeff[g] = [np.zeros((len(states[g]), N, N)) for i in attn_layers]
        for i, s in enumerate(states[g]):

            for j, layer in enumerate(attn_layers):
                attn_coeff[g][j][i, :, :] = layer.predict([s, adj], steps=1)

            s = s.reshape(adj.shape[0], 1)
            for layer in range(len(attn_layers)):
                for in_s in state_label:
                    for out_s in state_label:
                        in_l, out_l = state_label[in_s], state_label[out_s]
                        avail_s = (s==in_l) * (s==out_l).T * adj
                        a = attn_coeff[g][layer][i, avail_s==1]
                        cond_attn[layer][(in_s, out_s)].extend(a)

            p_bar.update()
    p_bar.close()

    return attn_coeff, cond_attn


def compute_loss(dynamics, model, states, graphs):
    N = model.num_nodes
    num_states = np.sum([states[g].shape[0] for g in graphs])
    state_label = dynamics.state_label

    losses_from_data = {}
    losses_from_dynamics = {}
    losses_from_data_per_degree = {k: [] for k in range(N)}
    losses_from_dynamics_per_degree = {k: [] for k in range(N)}
    loss = lambda x, y: - np.sum(x * np.log(y), axis=1)

    p_bar = tqdm.tqdm(range(num_states), "Losses")
    for g in graphs:
        dynamics.graph = nx.from_numpy_array(graphs[g])

        losses_from_data[g] = np.zeros(states[g].shape)
        losses_from_dynamics[g] = np.zeros(states[g].shape)

        degrees = np.sum(graphs[g], axis=0)

        for i, s in enumerate(states[g]):
            dynamics_prob = dynamics.predict(s)
            model_prob = model.model.predict([s, graphs[g]], steps=1)
            ss = np.zeros((s.shape[0], len(state_label)), dtype="int")
            ss[np.arange(s.shape[0]), s.astype("int")] = 1
            losses_from_data[g][i, :] = loss(ss, model_prob)
            losses_from_dynamics[g][i, :] = loss(dynamics_prob, model_prob)
            p_bar.update()

        for k in range(int(np.min(degrees)), int(np.max(degrees) + 1)):
            losses_from_data_per_degree[k].extend(list(losses_from_dynamics[g][i, :][degrees==k]))
            losses_from_dynamics_per_degree[k].extend(list(losses_from_data[g][i, :][degrees==k]))

    avg_loss_from_data = np.zeros(N)
    var_loss_from_data = np.zeros(N)
    avg_loss_from_dynamics = np.zeros(N)
    var_loss_from_dynamics = np.zeros(N)
    # Coarse-graining losses per degree
    for k in range(N):
        data_num = len(losses_from_data_per_degree[k])
        dynamics_num = len(losses_from_data_per_degree[k])
        if data_num > 0:
            avg_loss_from_data[k] = np.mean(losses_from_data_per_degree[k])
            var_loss_from_data[k] = np.std(losses_from_data_per_degree[k]) / np.sqrt(data_num)
        if dynamics_num > 0:
            avg_loss_from_dynamics[k] = np.mean(losses_from_dynamics_per_degree[k])
            var_loss_from_dynamics[k] = np.std(losses_from_dynamics_per_degree[k]) / np.sqrt(dynamics_num)

    p_bar.close()

    return losses_from_data, losses_from_dynamics,\
           avg_loss_from_data, var_loss_from_data,\
           avg_loss_from_dynamics, var_loss_from_dynamics


def main():
    prs = ap.ArgumentParser(description="Get local transition probability \
                                         figure from path to parameters.")
    prs.add_argument('--path_to_param', '-p', type=str, required=True,
                     help='Path to parameters.')
    prs.add_argument('--num_sample', '-n', default=-1,
                     help='Number of samples to use.')
    if len(sys.argv) == 1:
        prs.print_help()
        sys.exit(1)
    args = prs.parse_args()
    with open(args.path_to_param, 'r') as f:
        params = json.load(f)


    num_sample = int(args.num_sample)
    experiment = u.get_experiment(params, False)

    h5file = h5py.File(os.path.join(params["path"], params["name"] + "_model.h5"), 'r')
    experiment.load_hdf5_model(h5file)
    h5file.close()

    h5file = h5py.File(os.path.join(params["path"], params["name"] + "_data.h5"), 'r')
    experiment.load_hdf5_data(h5file)
    h5file.close()

    graphs = experiment.data_generator.graph_inputs
    states = {}
    targets = {}
    for g in graphs:
        states[g] = experiment.data_generator.state_inputs[g][:num_sample]
        targets[g] = experiment.data_generator.targets[g][:num_sample]

    model = experiment.model
    dynamics = experiment.data_generator.dynamics_model

    N = model.num_nodes
    rel_trans = params["dynamics"]["params"]["relevant_transitions"]

    an_h5file = h5py.File(os.path.join(params["path"],
                                       params["name"] + "_analytics.h5"),
                          'w')

    # Getting transition probability
    dynamics_tp, model_tp = compute_transition_probability(dynamics,
                                                           model,
                                                           states,
                                                           graphs)

    # Saving transition probabilities
    for g in graphs:
        an_h5file.create_dataset("/analytics/trans_prob/" + g + "/ground_truth", data=dynamics_tp[g])
        an_h5file.create_dataset("/analytics/trans_prob/" + g + "/model", data=model_tp[g])



    # Getting Local transition probabilities
    ltp_data = compute_ltp(dynamics, model, states, targets, graphs, rel_trans)
    for t in rel_trans:
        in_s, out_s = t[0], t[1]

        group = "/analytics/local_trans_prob/ground_truth/" + in_s + "_to_" + out_s
        val = ltp_data[0][(in_s, out_s)]
        an_h5file.create_dataset(group, data=val)

        group = "/analytics/local_trans_prob/model/" + in_s + "_to_" + out_s
        avg, var = ltp_data[1][(in_s, out_s)], ltp_data[3][(in_s, out_s)] 
        an_h5file.create_dataset(group, data=np.vstack((avg, var)).T)

        group = "/analytics/local_trans_prob/estimate/" + in_s + "_to_" + out_s
        avg, var = ltp_data[2][(in_s, out_s)], ltp_data[4][(in_s, out_s)]
        an_h5file.create_dataset(group, data=np.vstack((avg, var)).T)

    # Getting Local transition probabilities of star graph
    ltp_data = compute_star_ltp(dynamics, model, rel_trans)
    for t in rel_trans:
        in_s, out_s = t[0], t[1]

        group = "/analytics/star_ltp/ground_truth/" + in_s + "_to_" + out_s
        val = ltp_data[0][(in_s, out_s)]
        an_h5file.create_dataset(group, data=val)

        group = "/analytics/star_ltp/model/" + in_s + "_to_" + out_s
        val = ltp_data[1][(in_s, out_s)]
        an_h5file.create_dataset(group, data=val)

    # Getting loss per node
    loss_data = compute_loss(dynamics, model, states, graphs)
    for g in graphs:
        an_h5file.create_dataset("/analytics/losses/" + g + "/from_data",
                              data=loss_data[0][g])
        an_h5file.create_dataset("/analytics/losses/" + g + "/from_dynamics",
                              data=loss_data[1][g])
        an_h5file.create_dataset("/analytics/losses/from_data_per_degree",
                              data=np.vstack((loss_data[2], loss_data[3])).T)
        an_h5file.create_dataset("/analytics/losses/from_dynamics_per_degree",
                              data=np.vstack((loss_data[4], loss_data[5])).T)

    # Getting attention coefficients
    attn_layers = u.get_all_attn_layers(model)
    attn_coeff, cond_attn = compute_attn_coeff(dynamics, model, attn_layers,
                                               states, graphs)
    # Saving attention coefficients
    for g in graphs:
        for layer in range(len(attn_layers)):
            group = "/analytics/attn_coeff/layer"  + str(layer) + "/" + g
            an_h5file.create_dataset(group, data=attn_coeff[g][layer])

    for in_s in dynamics.state_label:
        for out_s in dynamics.state_label:
            for layer in range(len(attn_layers)):
                group = "/analytics/attn_coeff/layer" + str(layer) + "/" + in_s + "_to_" + out_s
                an_h5file.create_dataset(group, data=cond_attn[layer][(in_s, out_s)])


    if N <= 15:
        # Compute Markov matrices and occurence
        dynamics_markovmat, model_markovmat = compute_markov_matrix(dynamics,
                                                                    model,
                                                                    states,
                                                                    graphs)
        for g in graphs:
            group = "/analytics/markovmatrix/" + g + "/ground_truth/"
            an_h5file.create_dataset(group, data=dynamics_markovmat[g])
            group = "/analytics/markovmatrix/" + g + "/model/"
            an_h5file.create_dataset(group, data=model_markovmat[g])

        # Compute state occurence
        occurence = compute_state_occurence(dynamics, model, states, graphs)
        for g in graphs:
            group = "/analytics/occurence/" + g
            an_h5file.create_dataset(group, data=occurence[g])

    an_h5file.close()

if __name__ == '__main__':
    main()
