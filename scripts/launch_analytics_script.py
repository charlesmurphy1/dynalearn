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



with open('parameters.json', 'r') as f:
    params = json.load(f)


num_sample = 100
experiment = u.get_experiment(params)

h5file = h5py.File(os.path.join(params["path"],
                                params["experiment_name"] + ".h5"), 'r')

experiment.load_hdf5_model(h5file)
experiment.load_hdf5_data(h5file)

graphs = experiment.data_generator.graph_inputs
states = experiment.data_generator.state_inputs
targets = experiment.data_generator.targets

model = experiment.model
dynamics = experiment.data_generator.dynamics_model


# Getting transition probability
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
    for i, s in enumerate(states[g][:num_sample]):
        dynamics_tp[g][i, :, :] = dynamics.predict(s)
        model_tp[g][i, :, :] = model.model.predict([s, graphs[g]], steps=1)

# Getting local transition probabilitymodel_ltp = {}
dynamics_ltp = {}
model_ltp = {}
estimate_ltp = {}
for g in graphs:
    dynamics.graph = nx.from_numpy_array(graphs[g])
    in_s = states[g][:num_sample]
    out_s = targets[g][:num_sample, :]
    dynamics_ltp[g] = dynamics.ltp(in_s)
    model_ltp[g] = dynamics.model_ltp(model.model, in_s)
    estimate_ltp[g] = dynamics.estimate_ltp(in_s, out_s)


# Getting attention coefficients
attn_layers = u.get_all_attn_layers(model)
cond_attn_coeff = {l:{(in_s, out_s): [] for in_s in dynamics.state_label
                                     for out_s in dynamics.state_label}
                    for l in range(len(attn_layers))}
for g in graphs:
    adj = graphs[g]
    for i, s in enumerate(states[g][:num_sample]):
        if i%100 == 0: print(i)

        attn_coeff = []

        for i, layer in enumerate(attn_layers):
            attn_coeff.append(layer.predict([s, adj], steps=1))

        s = s.reshape(adj.shape[0], 1)
        for in_s, in_l in dynamics.state_label.items():
            for out_s, out_l in dynamics.state_label.items():
                avail_s = (s==in_l) * (s==out_l).T * adj
                for layer in range(len(attn_layers)):
                    a = attn_coeff[layer][avail_s==1]
                    cond_attn_coeff[layer][(in_s, out_s)].extend(a)

h5file.close()

# Writting analytics to file
h5file = h5py.File(os.path.join(params["path"],
                                params["experiment_name"] + "_analytics.h5"),
                   'w')


for g in graphs:
    # Saving transition probabilities
    h5file.create_dataset("/analytics/trans_prob/" + g + "/ground_truth", data=dynamics_tp[g])
    h5file.create_dataset("/analytics/trans_prob/" + g + "/model", data=model_tp[g])

    # Saving local transition probabilities
    for in_s in dynamics.state_label:
        for out_s in dynamics.state_label:
            group = "/analytics/local_trans_prob/" + g + "/ground_truth/" + in_s + "_to_" + out_s
            val = dynamics_ltp[g][(in_s, out_s)]
            h5file.create_dataset(group, data=val)

            group = "/analytics/local_trans_prob/" + g + "/model/" + in_s + "_to_" + out_s
            avg, var = model_ltp[g][0][(in_s, out_s)], model_ltp[g][1][(in_s, out_s)]
            h5file.create_dataset(group, data=np.vstack((avg, var)).T)

            group = "/analytics/local_trans_prob/" + g + "/estimate/" + in_s + "_to_" + out_s
            avg, var = estimate_ltp[g][0][(in_s, out_s)], estimate_ltp[g][1][(in_s, out_s)]
            h5file.create_dataset(group, data=np.vstack((avg, var)).T)

# Saving attention coefficients
for in_s in dynamics.state_label:
    for out_s in dynamics.state_label:
        for layer in range(len(attn_layers)):
            group = "/analytics/attention_coeff_" + str(layer) + "/" + in_s + "_to_" + out_s
            h5file.create_dataset(group, data=cond_attn_coeff[layer][(in_s, out_s)])