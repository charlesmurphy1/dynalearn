import numpy as np
from abc import ABC, abstractmethod
import copy


class Sampler(ABC):
    def __init__(self, name, config, verbose=0):
        self.name = name
        self.__config = config

        self.sample_from_weights = config.sample_from_weights
        self.resample = config.resample
        self.iteration = 0
        self.sampling_bias = 1

        self.num_nodes = dict()
        self.num_samples = dict()

        self.graph_set = list()
        self.avail_graph_set = list()
        self.graph_weights = dict()

        self.state_set = dict()
        self.avail_state_set = dict()
        self.state_weights = dict()

        self.node_set = dict()
        self.avail_node_set = dict()
        self.node_weights = dict()

        self.verbose = verbose

    def __call__(self, batch_size):

        g_index = self.get_graph()
        t_index = self.get_state(g_index)
        n_index = self.get_nodes(g_index, t_index, batch_size)
        self.iteration += 1
        if self.iteration == self.resample:
            self.reset_set()

        return g_index, t_index, n_index

    def copy(self):
        sampler_copy = self.__class__(self.name, self.__config, verbose=self.verbose)

        sampler_copy.num_nodes = self.num_nodes
        sampler_copy.num_samples = self.num_samples

        sampler_copy.graph_set = copy.deepcopy(self.graph_set)
        sampler_copy.avail_graph_set = copy.deepcopy(self.avail_graph_set)
        sampler_copy.graph_weights = copy.deepcopy(self.graph_weights)

        sampler_copy.state_set = copy.deepcopy(self.state_set)
        sampler_copy.avail_state_set = copy.deepcopy(self.avail_state_set)
        sampler_copy.state_weights = copy.deepcopy(self.state_weights)

        sampler_copy.node_set = copy.deepcopy(self.node_set)
        sampler_copy.avail_node_set = copy.deepcopy(self.avail_node_set)
        sampler_copy.node_weights = copy.deepcopy(self.node_weights)

        return sampler_copy

    def update(self, graphs, inputs):
        for g in graphs:
            adj = graphs[g]
            self.num_samples[g] = inputs[g].shape[0]
            self.num_nodes[g] = inputs[g].shape[1]
            self.node_set[g] = dict()
            self.avail_node_set[g] = dict()

            for t in range(self.num_samples[g]):
                self.node_set[g][t] = np.arange(self.num_nodes[g]).astype("int")
                self.avail_node_set[g][t] = np.arange(self.num_nodes[g]).astype("int")

            self.state_set[g] = list(range(self.num_samples[g]))
            self.avail_state_set[g] = list(range(self.num_samples[g]))

        self.graph_set = list(graphs.keys())
        self.avail_graph_set = list(graphs.keys())
        self.update_weights(graphs, inputs)
        # print(self.name, self.avail_node_set["BAGraph_0"][0])

        return

    def reset_set(self):
        self.avail_graph_set = self.graph_set.copy()
        for n in self.graph_set:
            self.avail_state_set[n] = self.state_set[n].copy()

    def update_set(self, g_index):
        if len(self.avail_state_set[g_index]) == 0:
            self.avail_graph_set.remove(g_index)
            if len(self.avail_graph_set) == 0:
                self.reset_set()

    def get_nodes(self, g_index, s_index, batch_size=-1):
        mask = np.zeros(self.num_nodes[g_index])
        p = self.node_weights[g_index][s_index] ** self.sampling_bias
        if np.sum(p) > 0:
            p /= np.sum(p)
        else:
            p = None
        if self.sample_from_weights:
            if batch_size == -1 or batch_size > len(
                self.avail_node_set[g_index][s_index]
            ):
                mask[self.avail_node_set[g_index][s_index]] = 1
            else:
                n_index = np.random.choice(
                    self.node_set[g_index][s_index], size=batch_size, p=p, replace=False
                ).astype("int")
                mask[n_index] = 1
        else:
            if batch_size == -1 or batch_size > len(
                self.avail_node_set[g_index][s_index]
            ):
                p = p[self.avail_node_set[g_index][s_index]]
                mask[self.avail_node_set[g_index][s_index]] = p * np.sum(p > 0)
            else:
                n_index = np.random.choice(
                    self.node_set[g_index][s_index], size=batch_size, p=p, replace=False
                ).astype("int")
                p = p[n_index]
                mask[n_index] = p * np.sum(p > 0)
        return mask

    def sample_nodes(self, g_index, s_index, size=None, bias=1):
        mask = np.zeros(self.num_nodes[g_index])
        if size is None or size > len(self.avail_node_set[g_index][s_index]):
            mask[self.avail_node_set[g_index][s_index]] = 1
        else:
            p = self.node_weights[g_index][s_index] ** (bias)
            p /= np.sum(p)
            n_index = np.random.choice(
                self.node_set[g_index][s_index], size=size, p=p, replace=False
            ).astype("int")
            mask[n_index] = 1
        return np.where(mask == 1)[0]

    def update_weights(self, graphs, inputs):
        self.node_weights = dict()
        self.state_weights = dict()
        self.graph_weights = dict()

        for g in graphs:
            self.node_weights[g] = dict()
            self.state_weights[g] = dict()

            for t, s in enumerate(inputs[g]):
                self.node_weights[g][t] = np.zeros(self.num_nodes[g])
                self.node_weights[g][t][self.avail_node_set[g][t]] = 1
                self.state_weights[g][t] = np.sum(self.node_weights[g][t])

            self.graph_weights[g] = np.sum(
                [self.state_weights[g][int(t)] for t in self.avail_state_set[g]]
            )

    def save(self, h5file, overwrite=False):
        if "sampler/" + self.name in h5file:
            if overwrite:
                del h5file["sampler/" + self.name]
            else:
                return
        h5file.create_group("sampler/" + self.name)
        h5group = h5file["sampler/" + self.name]
        for g in self.graph_set:
            avail_node_set = np.array(
                [
                    np.where(
                        self.node_weights[g][t] > 0,
                        np.ones(self.node_weights[g][t].shape),
                        np.zeros(self.node_weights[g][t].shape),
                    )
                    for t in range(self.num_samples[g])
                ]
            )
            node_weights = np.array(
                [self.node_weights[g][t] for t in range(self.num_samples[g])]
            )
            state_weights = np.array(
                [self.state_weights[g][t] for t in range(self.num_samples[g])]
            )
            graph_weights = self.graph_weights[g]
            h5group.create_dataset(g + "/avail_node_set", data=avail_node_set)
            h5group.create_dataset(g + "/node_weights", data=node_weights)
            h5group.create_dataset(g + "/state_weights", data=state_weights)
            h5group.create_dataset(g + "/graph_weights", data=graph_weights)

    def load(self, h5file):
        if "sampler/" + self.name in h5file:
            for k, v in h5file["sampler/" + self.name].items():
                avail_node_set = v["avail_node_set"][...]
                node_weights = v["node_weights"][...]
                state_weights = v["state_weights"][...]
                graph_weights = v["graph_weights"][...]

                self.num_samples[k] = node_weights.shape[0]
                self.num_nodes[k] = node_weights.shape[1]

                if k not in self.graph_set or k not in self.avail_graph_set:
                    self.graph_set.append(k)
                    self.avail_graph_set.append(k)
                self.graph_weights[k] = graph_weights

                self.state_set[k] = list(range(self.num_samples[k]))
                self.avail_state_set[k] = list(range(self.num_samples[k]))
                self.state_weights[k] = {
                    i: weights for i, weights in enumerate(state_weights)
                }

                self.node_set[k] = {
                    t: np.arange(self.num_nodes[k]).astype("int")
                    for t in range(self.num_samples[k])
                }
                self.avail_node_set[k] = {
                    t: np.argwhere(nodes) for t, nodes in enumerate(avail_node_set)
                }
                self.node_weights[k] = {
                    i: weights for i, weights in enumerate(node_weights)
                }

    @abstractmethod
    def get_graph(self):
        raise NotImplementedError()

    @abstractmethod
    def get_state(self, g_index):
        raise NotImplementedError()
