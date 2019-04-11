import dynalearn as dl
import h5py
import numpy as np


# Graph parameters

np_seed = 1
tf_seed = 2
np.random.seed(np_seed)
tf.set_random_seed(tf_seed)

N = 100
avgk = 5
graph_model = dl.graphs.ERGraph(N, avgk / N, np_seed)

# Dynamics parameters
inf_prob = 0.04
rec_prob = 0.08
init_state = None
dynamics_model = dl.dynamics.SISDynamics(inf_prob, rec_prob, init_state)

# Data generator parameters
batch_size = 32
shuffle = True
prohibited_node_index=[],
max_null_iter=100
T = 200
num_sample = 100000
gamma = 0.
data_generator = dl.generators.MarkovBinaryDynamicsGenerator(graph_model, dynamics_model,
                                                             batch_size,
                                                             shuffle=shuffle, 
                                                             prohibited_node_index=prohibited_node_index,
                                                             max_null_iter=max_null_iter)

# Model parameters
n_hidden = [32, 32]
n_heads = 6
wd = 1e-4
dropout = 0.
model = dl.models.GATMarkovBinaryPredictor(N,
                                           n_hidden,
                                           n_heads,
                                           weight_decay=wd,
                                           dropout=dropout,
                                           seed=tf_seed)

# Trainer parameters
loss = keras.losses.binary_crossentropy
optimizer = keras.optimizers.Adam
metrics = ['accuracy']
learning_rate=1e-4
callbacks = []

exp = dl.Experiment("cedar_test", model, data_generator,
                    loss=loss,
                    optimizer=optimizer,
                    metrics=metrics,
                    learning_rate=learning_rate,
                    callbacks=callbacks,
                    numpy_seed=np_seed,
                    tensorflow_seed=tf_seed)

exp.generate_data(num_sample, T, gamma=gamma)
exp.train_model(50, 10000, verbose=1)
h5file = h5py.File('testcedar_experiment.h5', 'w')
exp.save_hdf5_all(h5file)
h5file.close()
