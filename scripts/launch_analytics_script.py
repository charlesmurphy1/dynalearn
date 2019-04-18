import dynalearn as dl
import h5py
import json
import networkx as nx
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K
import utilities as u
import tqdm



with open('parameters.json', 'r') as f:
    params = json.load(f)


num_sample = 4000
experiment = u.get_experiment(params)

h5file = h5py.File(os.path.join(params["path"],
                                params["experiment_name"] + ".h5"), 'r')

experiment.load_hdf5_model(h5file)
experiment.load_hdf5_data(h5file)

graphs = {"ERGraph_0":experiment.data_generator.graph_inputs["ERGraph_0"]}
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
# Getting local transition probability
p_bar = tqdm.tqdm(range(len(graphs)), "Local transition probabilities")
avg_dynamics_ltp = {(in_s, out_s): np.zeros(N) for in_s in state_label
                                               for out_s in state_label}

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
    for in_s in state_label:
        for out_s in state_label:
            avg_dynamics_ltp[(in_s, out_s)] = dynamics_ltp[(in_s, out_s)]
            avg_model_ltp[(in_s, out_s)] = model_ltp[0][(in_s, out_s)]
            avg_estimate_ltp[(in_s, out_s)] = estimate_ltp[0][(in_s, out_s)]

            var_model_ltp[(in_s, out_s)] = model_ltp[1][(in_s, out_s)]
            var_estimate_ltp[(in_s, out_s)] = estimate_ltp[1][(in_s, out_s)]
    p_bar.update()
p_bar.close()



# Getting attention coefficients
p_bar = tqdm.tqdm(range(num_states), "Attention coefficients")
attn_layers = u.get_all_attn_layers(model)
cond_attn_coeff = {l:{(in_s, out_s): [] for in_s in dynamics.state_label
                                     for out_s in dynamics.state_label}
                    for l in range(len(attn_layers))}
for g in graphs:
    adj = graphs[g]
    for i, s in enumerate(states[g]):

        attn_coeff = []

        for j, layer in enumerate(attn_layers):
            attn_coeff.append(layer.predict([s, adj], steps=1))

        s = s.reshape(adj.shape[0], 1)
        for in_s, in_l in dynamics.state_label.items():
            for out_s, out_l in dynamics.state_label.items():
                avail_s = (s==in_l) * (s==out_l).T * adj
                for layer in range(len(attn_layers)):
                    a = attn_coeff[layer][avail_s==1]
                    cond_attn_coeff[layer][(in_s, out_s)].extend(a)
        p_bar.update()
p_bar.close()

h5file.close()

# Writting analytics to file
h5file = h5py.File(os.path.join(params["path"],
                                params["experiment_name"] + "_analytics.h5"), 'w')

for g in graphs:
    # Saving transition probabilities
    h5file.create_dataset("/analytics/trans_prob/" + g + "/ground_truth", data=dynamics_tp[g])
    h5file.create_dataset("/analytics/trans_prob/" + g + "/model", data=model_tp[g])

for in_s in state_label:
    for out_s in state_label:

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

        # Saving attention coefficients
        for layer in range(len(attn_layers)):
            group = "/analytics/attention_coeff_" + str(layer) + "/" + in_s + "_to_" + out_s
            h5file.create_dataset(group, data=cond_attn_coeff[layer][(in_s, out_s)])
