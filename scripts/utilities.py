import dynalearn as dl
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tqdm

from dynalearn.models.layers import GraphAttention


def get_noisy_crossentropy(noise=0):

    def noisy_crossentropy(y_true, y_pred):
        num_classes = tf.cast(K.shape(y_true)[1], tf.float32)
        y_true = y_true * (1 - noise) +\
                 (1 - y_true) * noise / num_classes

        return keras.losses.categorical_crossentropy(y_true, y_pred)

    return noisy_crossentropy





def get_graph(graph_name, params):
    if 'CycleGraph' == graph_name:
        return dl.graphs.CycleGraph(params["graph"]["params"]['N'],
                                    params["np_seed"])
    elif 'CompleteGraph' == graph_name:
        return dl.graphs.CompleteGraph(params["graph"]["params"]['N'],
                                       params["np_seed"])
    elif 'StarGraph' == graph_name:
        return dl.graphs.StarGraph(params["graph"]["params"]['N'],
                                   params["np_seed"])
    elif 'EmptyGraph' == graph_name:
        return dl.graphs.EmptyGraph(params["graph"]["params"]['N'],
                                    params["np_seed"])
    elif 'RegularGraph' == graph_name:
        return dl.graphs.RegularGraph(params["graph"]["params"]['N'],
                                      params["graph"]["params"]['degree'],
                                      params["np_seed"])
    elif 'ERGraph' == graph_name:
        return dl.graphs.ERGraph(params["graph"]["params"]['N'],
                                 params["graph"]["params"]['p'],
                                 params["np_seed"])
    elif 'BAGraph' == graph_name:
        return dl.graphs.BAGraph(params["graph"]["params"]['N'],
                                 params["graph"]["params"]['M'],
                                 params["np_seed"])
    else:
        raise ValueError('wrong string name for graph.')


def get_dynamics(dynamics_name, params):
    if 'SISDynamics' == dynamics_name:
        if params["dynamics"]["params"]['init_param'] == "None":
            params["dynamics"]["params"]['init_param'] = None
        return dl.dynamics.SISDynamics(params["dynamics"]["params"]['infection_prob'],
                                       params["dynamics"]["params"]['recovery_prob'],
                                       params["dynamics"]["params"]['init_param']
                                       )
    elif 'SIRDynamics' == dynamics_name:
        if params["dynamics"]["params"]['init_param'] == "None":
            params["dynamics"]["params"]['init_param'] = None
        return dl.dynamics.SIRDynamics(params["dynamics"]["params"]['infection_prob'],
                                       params["dynamics"]["params"]['recovery_prob'],
                                       params["dynamics"]["params"]['init_param'])
    else:
        raise ValueError('wrong string name for dynamics.')

        
def get_model(model_name, params):
    if 'GATMarkovBinaryPredictor' == model_name:
        return dl.models.GATMarkovBinaryPredictor(params["graph"]["params"]["N"],
                                                  params["dynamics"]["params"]["num_states"],
                                                  params["model"]["params"]["n_hidden"], 
                                                  params["model"]["params"]["n_heads"],
                                                  weight_decay=params["model"]["params"]["weight_decay"],
                                                  dropout=params["model"]["params"]["dropout"],
                                                  seed=params['tf_seed'])
    else:
        raise ValueError('wrong string name for model.')


def get_datagenerator(gen_name, graph_model, dynamics_model, params):
    if 'MarkovBinaryDynamicsGenerator' == gen_name:
        return dl.generators.MarkovBinaryDynamicsGenerator(graph_model, dynamics_model,
                                                           params["data_generator"]["params"]["batch_size"],
                                                           shuffle=True, 
                                                           prohibited_node_index=[],
                                                           max_null_iter=params["data_generator"]["params"]["max_null_iter"])
    else:
        raise ValueError('wrong string name for data generator.')


def get_experiment(params):
    # Define seeds
    np.random.seed(params["np_seed"])
    tf.set_random_seed(params["tf_seed"])

    # Define graph
    graph = get_graph(params["graph"]["name"], params)

    # Define dynamics
    dynamics = get_dynamics(params["dynamics"]["name"], params)

    # Define data generator 
    data_generator = get_datagenerator(params["data_generator"]["name"],
                                       graph,
                                       dynamics,
                                       params)

    # Define model

    model = get_model(params["model"]["name"], params)
    model.model.summary()
    optimizer = keras.optimizers.get(params["training"]["optimizer"])
    if params["training"]["loss"] == "noisy_crossentropy":
        loss = get_noisy_crossentropy(noise=params["training"]["target_noise"])
    else:
        loss = keras.losses.get(params["training"]["loss"])
    schedule = get_schedule(params["training"]["schedule"])
    metrics = ["accuracy"]
    callbacks = [keras.callbacks.LearningRateScheduler(schedule, verbose=1)]
    # if params["training"]["with_mutual_info_metrics"]:
    #     temp_generator = get_datagenerator(params["data_generator"]["name"],
    #                                        graph,
    #                                        dynamics,
    #                                        params)
    #     temp_generator.generate(5000, 5000)
    #     for g in temp_generator.graph_inputs: 
    #         adj = temp_generator.graph_inputs[g]
    #         states = temp_generator.state_inputs[g]
    #     cb = MutualInformationMetrics(states, adj, 1, params["dynamics"]["params"]["num_states"])
    #     callbacks.append(cb)


    # Define experiment
    experiment = dl.Experiment(params["experiment_name"],
                               model,
                               data_generator,
                               loss=loss,
                               optimizer=optimizer,
                               metrics=metrics,
                               learning_rate=params["training"]["learning_rate"],
                               callbacks=callbacks,
                               numpy_seed=params["np_seed"],
                               tensorflow_seed=params["tf_seed"])


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
    return int(np.sum(x * base**np.arange(len(x))))

def bin_ts(ts, delay, num_states):
    delay += 1
    states = np.zeros((num_states**delay, delay))
    counts = np.zeros(num_states**delay)

    for i in range(1, num_states**delay):
        states[i] = increment_int_from_base(states[i - 1], num_states)

    for t in range(len(ts) - delay + 1):
        _ts = ts[t:t+delay]
        i = np.where(np.all(states == _ts, axis=1))[0][0]
        counts[i] += 1
    return counts, states

def bin_ts1_ts2(ts1, ts2, delay, num_states):
    delay += 1
    states = np.zeros((num_states**(2 * delay), 2 * delay))
    counts = np.zeros(num_states**(2 * delay))

    s1 = np.zeros(delay)
    for i in range(num_states**(delay)):
        s2 = np.zeros(delay)
        for j in range(num_states**(delay)):
            states[i * num_states**(delay) + j, :delay] = s1
            states[i * num_states**(delay) + j, delay:] = s2
            s2 = increment_int_from_base(s2, num_states)
        s1 = increment_int_from_base(s1, num_states)

    for t in range(len(ts1) - delay + 1):
        _ts1 = ts1[t:t+delay]
        _ts2 = ts2[t:t+delay]
        _ts = np.concatenate((_ts1, _ts2))
        i = np.where(np.all(states == _ts, axis=1))[0][0]
        counts[i] += 1

    return counts, states


def information(ts, delay, num_states):
    counts, states = bin_ts(ts, delay, num_states)
    marginal_prob = counts / np.sum(counts)
    info = 0
    for i in range(num_states**delay):
        marg = marginal_prob[i]
        if marg > 0 :
            info -= marg * np.log(marg)

    return info


def mutual_information(ts1, ts2, delay, num_states):
    counts1, states1 = bin_ts(ts1, delay, num_states)
    counts2, states2 = bin_ts(ts2, delay, num_states)
    counts12, states12 = bin_ts1_ts2(ts1, ts2, delay, num_states)
    marginal_prob_1 = counts1 / np.sum(counts1)
    marginal_prob_2 = counts2 / np.sum(counts2)
    joint_prob = counts12 / np.sum(counts12)

    mutual_info = 0

    for i in range(num_states**delay):
        marg1 = marginal_prob_1[i]
        for j in range(num_states**delay):
            marg2 = marginal_prob_2[j]
            joint = joint_prob[i * num_states**delay + j]
            if joint > 0 and marg1 > 0 and marg2 > 0:
                mutual_info += joint * np.log(joint / marg1 / marg2)
    return mutual_info


class MutualInformationMetrics(keras.callbacks.Callback):
    def __init__(self, states, adj, delay, num_states, verbose=1):
        super(MutualInformationMetrics, self).__init__()
        self.adj = adj
        self.states = states
        self.N = self.adj.shape[0]
        self.num_samples = self.states.shape[0]
        self.delay = delay
        self.num_states = num_states
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        self._data = {}

    def on_epoch_end(self, epoch, logs={}):
        info = np.zeros(self.N)
        mutual_info = np.zeros(self.N)
        predictions = np.array([self.model.predict([s, self.adj], steps=1)
                                for s in self.states])
        y_true = self.states
        y_pred = np.zeros((self.num_samples, self.N))
        if self.verbose: p_bar = tqdm.tqdm(range(self.N * self.num_samples), desc="Computing y_pred")
        for i in range(self.num_samples):
            for j in range(self.N):
                if self.verbose: p_bar.update()
                new_state = np.random.choice(range(self.num_states),
                                             p=predictions[i, j, :])
                y_pred[i, j] = new_state
        if self.verbose:
            p_bar.close()
        
        if self.verbose: p_bar = tqdm.tqdm(range(self.N), desc="Computing MI")

        for i in range(self.N):
            info[i] = information(y_true[:, i], self.delay, self.num_states)
            mutual_info[i] = mutual_information(y_true[:, i], y_pred[:, i],
                                                self.delay,
                                                self.num_states)
            if self.verbose: p_bar.update()
        if self.verbose:
            p_bar.close()
            print("Epoch {0}\t Avg. information: {1}\t Avg. mutual information: {2}".format(epoch, np.mean(info), np.mean(mutual_info)))

        self._data[epoch] = np.array([info, mutual_info]).T

        return

    def get_data(self):
        return self._data
        


def get_all_attn_layers(model):

    attn_layers = []
    num_attn = 0
    for layer in model.model.layers:
        if type(layer) == GraphAttention:
            num_attn += 1
            attn_layers.append(model.get_attn_layer(num_attn))

    return attn_layers



