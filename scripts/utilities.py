import dynalearn as dl
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tqdm

from dynalearn.models.layers import GraphAttention


def get_noisy_crossentropy(noise=0):
    def noisy_crossentropy(y_true, y_pred):
        num_classes = tf.cast(K.shape(y_true)[1], tf.float32)
        y_true = y_true * (1 - noise) + (1 - y_true) * noise / num_classes

        return keras.losses.categorical_crossentropy(y_true, y_pred)

    return noisy_crossentropy


def get_graph(params):
    if "CycleGraph" == params["graph"]["name"]:
        return dl.graphs.CycleGraph(params["graph"]["params"]["N"])
    elif "CompleteGraph" == params["graph"]["name"]:
        return dl.graphs.CompleteGraph(params["graph"]["params"]["N"])
    elif "StarGraph" == params["graph"]["name"]:
        return dl.graphs.StarGraph(params["graph"]["params"]["N"])
    elif "EmptyGraph" == params["graph"]["name"]:
        return dl.graphs.EmptyGraph(params["graph"]["params"]["N"])
    elif "RegularGraph" == params["graph"]["name"]:
        return dl.graphs.RegularGraph(
            params["graph"]["params"]["N"], params["graph"]["params"]["degree"]
        )
    elif "ERGraph" == params["graph"]["name"]:
        return dl.graphs.ERGraph(
            params["graph"]["params"]["N"], params["graph"]["params"]["p"]
        )
    elif "BAGraph" == params["graph"]["name"]:
        return dl.graphs.BAGraph(
            params["graph"]["params"]["N"], params["graph"]["params"]["M"]
        )
    else:
        raise ValueError("wrong string name for graph.")


def get_dynamics(params):
    if "SISDynamics" == params["dynamics"]["name"]:
        if params["dynamics"]["params"]["init_param"] == "None":
            params["dynamics"]["params"]["init_param"] = None
        return dl.dynamics.SISDynamics(
            params["dynamics"]["params"]["infection_prob"],
            params["dynamics"]["params"]["recovery_prob"],
            params["dynamics"]["params"]["init_param"],
        )
    elif "SIRDynamics" == params["dynamics"]["name"]:
        if params["dynamics"]["params"]["init_param"] == "None":
            params["dynamics"]["params"]["init_param"] = None
        return dl.dynamics.SIRDynamics(
            params["dynamics"]["params"]["infection_prob"],
            params["dynamics"]["params"]["recovery_prob"],
            params["dynamics"]["params"]["init_param"],
        )
    else:
        raise ValueError("wrong string name for dynamics.")


def get_model(params, dynamics):
    if "LocalStatePredictor" == params["model"]["name"]:
        return dl.models.LocalStatePredictor(
            params["graph"]["params"]["N"],
            len(dynamics.state_label),
            params["model"]["params"]["n_hidden"],
            params["model"]["params"]["n_heads"],
            weight_decay=params["model"]["params"]["weight_decay"],
            dropout=params["model"]["params"]["dropout"],
            seed=params["tf_seed"],
        )
    else:
        raise ValueError("wrong string name for model.")


def get_sampler(params, dynamics):
    if params["sampler"]["name"] == "SequentialSampler":
        return dl.generators.SequentialSampler(
            batch_size=params["sampler"]["params"]["batch_size"],
            sample_from_weight=params["sampler"]["params"]["sample_from_weight"],
            replace=params["sampler"]["params"]["replace"],
        )
    elif params["sampler"]["name"] == "RandomSampler":
        return dl.generators.RandomSampler(
            batch_size=params["sampler"]["params"]["batch_size"],
            replace=params["sampler"]["params"]["replace"],
        )
    elif params["sampler"]["name"] == "DegreeBiasedSampler":
        return dl.generators.DegreeBiasedSampler(
            sampling_bias=params["sampler"]["params"]["sampling_bias"],
            batch_size=params["sampler"]["params"]["batch_size"],
            replace=params["sampler"]["params"]["replace"],
        )

    elif params["sampler"]["name"] == "StateBiasedSampler":
        return dl.generators.StateBiasedSampler(
            dynamics,
            sampling_bias=params["sampler"]["params"]["sampling_bias"],
            batch_size=params["sampler"]["params"]["batch_size"],
            replace=params["sampler"]["params"]["replace"],
        )
    else:
        raise ValueError("wrong string name for sampler.")


def get_generator(graph_model, dynamics_model, sampler, params):
    if "with_truth" in params["training"]:
        with_truth = params["training"]["with_truth"]
    else:
        with_truth = False

    if "MarkovBinaryDynamicsGenerator" == params["generator"]["name"]:
        return dl.generators.MarkovBinaryDynamicsGenerator(
            graph_model, dynamics_model, shuffle=True, with_truth=with_truth
        )
    elif "DynamicsGenerator" == params["generator"]["name"]:
        return dl.generators.DynamicsGenerator(
            graph_model, dynamics_model, sampler, with_truth=False, verbose=1
        )
    else:
        raise ValueError("wrong string name for generator.")


def get_experiment(params):
    # Define seeds
    np.random.seed(params["np_seed"])
    tf.set_random_seed(params["tf_seed"])

    # Define graph
    graph = get_graph(params)

    # Define dynamics
    dynamics = get_dynamics(params)

    # Define data generator
    sampler = get_sampler(params, dynamics)
    generator = get_generator(graph, dynamics, sampler, params)

    # Define model
    model = get_model(params, dynamics)
    optimizer = keras.optimizers.get(params["training"]["optimizer"])
    if params["training"]["loss"] == "noisy_crossentropy":
        loss = get_noisy_crossentropy(noise=params["training"]["target_noise"])
    else:
        loss = keras.losses.get(params["training"]["loss"])
    schedule = get_schedule(params["training"]["schedule"])
    metrics = []
    callbacks = [
        keras.callbacks.LearningRateScheduler(schedule, verbose=1),
        keras.callbacks.ModelCheckpoint(
            os.path.join(
                params["path"], params["name"] + "_" + params["path_to_best"] + ".hdf5"
            ),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        ),
    ]

    # Define experiment
    experiment = dl.Experiment(
        params["name"],
        model,
        generator,
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
        learning_rate=params["training"]["learning_rate"],
        callbacks=callbacks,
        numpy_seed=params["np_seed"],
        tensorflow_seed=params["tf_seed"],
    )

    return experiment


def get_schedule(schedule):
    def lr_schedule(epoch, lr):
        if (epoch + 1) % schedule["epoch"] == 0:
            lr /= schedule["factor"]
        return lr

    return lr_schedule


def int_to_base(i, base, size=None):

    if i > 0:
        if size is None or size < np.floor(np.log(i) / np.log(base)) + 1:
            size = np.floor(np.log(i) / np.log(base)) + 1
    else:
        if size is None:
            size = 1

    return (i // base ** np.arange(size)) % base


def increment_int_from_base(x, base):
    val = x * 1
    for i in range(len(x)):
        val[i] += 1
        if val[i] > base - 1:
            val[i] = 0
        else:
            break

    return val


def base_to_int(x, base):
    return int(np.sum(x * base ** np.arange(len(x))))


# def bin_timeseries(ts, num_states):
#     N = ts.shape[1]
#     states = np.arange(num_states).reshape(num_states, 1)
#     counts = np.zeros((num_states, N))

#     for x in ts:
#         for i in range(num_states):
#             counts[i] += x==states[i]
#     return counts, states


# def bin_two_timeseries(ts1, ts2, num_states):
#     N = ts1.shape[1]
#     states = np.zeros((num_states**2, 2))
#     counts = np.zeros((num_states**2, N))
#     for i in range(num_states):
#         for j in range(num_states):
#             states[i * num_states + j, 0] = i
#             states[i * num_states + j, 1] = j

#     for x, y in zip(ts1, ts2):
#         x, y = x.reshape(len(x), 1), y.reshape(len(x), 1)
#         xy = np.concatenate((x, y), axis=1)
#         for i in range(num_states**2):
#             counts[i] += np.all(xy==states[i], axis=1)
#     return counts, states


# def information(ts, num_states):
#     if len(ts.shape) == 1:
#         ts = ts.reshape(ts.shape[0], 1)
#     N = ts.shape[1]
#     counts, states = bin_timeseries(ts, num_states)
#     marginal_prob = counts / np.sum(counts)
#     info = np.zeros(N)
#     for i in range(num_states):
#         marg = marginal_prob[i]
#         index = marg > 0
#         info[index] -= marg[index] * np.log2(marg[index])

#     return info


# def mutual_information(ts1, ts2, num_states):
#     if len(ts1.shape) == 1:
#         ts1 = ts1.reshape(ts1.shape[0], 1)
#     if len(ts2.shape) == 1:
#         ts2 = ts2.reshape(ts2.shape[0], 1)
#     N = ts1.shape[1]
#     counts1, states1 = bin_timeseries(ts1, num_states)
#     counts2, states2 = bin_timeseries(ts2, num_states)
#     counts12, states12 = bin_two_timeseries(ts1, ts2, num_states)
#     marginal_prob_1 = counts1 / np.sum(counts1, axis=0)
#     marginal_prob_2 = counts2 / np.sum(counts2, axis=0)
#     joint_prob = counts12 / np.sum(counts12, axis=0)

#     mutual_info = np.zeros(N)
#     for i in range(num_states):
#         marg1 = marginal_prob_1[i]
#         for j in range(num_states):
#             marg2 = marginal_prob_2[j]
#             joint = joint_prob[i * num_states + j]
#             index = joint > 0
#             mutual_info[index] += joint[index] * np.log2(joint[index] / marg1[index] / marg2[index])
#     return mutual_info


# class MutualInformationMetrics(keras.callbacks.Callback):
#     def __init__(self, states, adj, num_states, verbose=1):
#         super(MutualInformationMetrics, self).__init__()
#         self.adj = adj
#         self.states = states
#         self.N = self.adj.shape[0]
#         self.num_samples = self.states.shape[0]
#         self.num_states = num_states
#         self.verbose = verbose
#         self.session = K.get_session()

#     def on_train_begin(self, logs={}):
#         self._data = {}

#     def on_epoch_end(self, epoch, logs={}):
#         info = np.zeros(self.N)
#         mutual_info = np.zeros(self.N)
#         p = np.array([self.model.predict([s, self.adj], steps=1)
#                       for s in self.states])
#         dist = tf.distributions.Categorical(probs=p)
#         y_true = self.states
#         y_pred = dist.sample().eval(session=self.session)


#         info = information(y_true, self.num_states)
#         mutual_info = mutual_information(y_true, y_pred, self.num_states)
#         if self.verbose:
#             print("\n")
#             print("Epoch {0}\t Avg. information: {1}\t Avg. mutual information: {2}".format(epoch, np.mean(info), np.mean(mutual_info)))

#         self._data[epoch] = np.array([info, mutual_info]).T

#         return

#     def get_data(self):
#         return self._data


def get_all_attn_layers(model):

    attn_layers = []
    num_attn = 0
    for layer in model.model.layers:
        if type(layer) == GraphAttention:
            num_attn += 1
            attn_layers.append(model.get_attn_layer(num_attn))

    return attn_layers
