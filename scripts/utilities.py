import dynalearn as dl
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from dynalearn.models.layers import GraphAttention


class noisy_crossentropy:
    def __init__(self, noise=0):
        self.noise = noise

    def __call__(self, y_true, y_pred):
        num_states = y_true.shape[1]
        y_true = y_true * (1 - self.noise) + \
                 (1 - y_true) * self.noise / num_states

        return keras.losses.categorical_crossentropy(y_true, y_pred)



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
    optimizer = keras.optimizers.get(params["training"]["optimizer"])
    # loss = keras.losses.binary_crossentropy
    if params["training"]["loss"] == "noisy_cross_entropy":
        loss = noisy_cross_entropy(params["training"]["target_noise"])
    else:
        loss = keras.losses.get(params["training"]["loss"])

    metrics = ["accuracy"]
    callbacks = []

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


def get_all_attn_layers(model):

    attn_layers = []
    num_attn = 0
    for layer in model.model.layers:
        if type(layer) == GraphAttention:
            num_attn += 1
            attn_layers.append(model.get_attn_layer(num_attn))

    return attn_layers



