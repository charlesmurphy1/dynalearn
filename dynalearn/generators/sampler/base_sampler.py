import numpy as np


class Sampler(object):
    def __init__(self, verbose=1, resample=-1):

        self.verbose = verbose
        self.resample = resample
        self.iteration = 0

        self.params = dict()
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

    def __call__(self, batch_size):

        g_index = self.get_graph()
        s_index = self.get_state(g_index)
        n_index = self.get_nodes(g_index, s_index, batch_size)
        self.iteration += 1
        if self.iteration == self.resample:
            self.reset_set()

        return g_index, s_index, n_index

    def update(self, graphs, inputs):
        for i in graphs:
            adj = graphs[i]
            self.num_samples[i] = inputs[i].shape[0]
            self.num_nodes[i] = inputs[i].shape[1]
            self.node_set[i] = dict()
            self.avail_node_set[i] = dict()

            for j in range(self.num_samples[i]):
                self.node_set[i][j] = np.arange(self.num_nodes[i]).astype("int")
                self.avail_node_set[i][j] = np.arange(self.num_nodes[i]).astype("int")

            self.state_set[i] = list(range(self.num_samples[i]))
            self.avail_state_set[i] = list(range(self.num_samples[i]))

        self.graph_set = list(graphs.keys())
        self.avail_graph_set = list(graphs.keys())
        self.update_weights(graphs, inputs)

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

    def get_graph(self):
        raise NotImplementedError()

    def get_state(self, g_index):
        raise NotImplementedError()

    def get_nodes(self, g_index, s_index, batch_size=-1):
        mask = np.zeros(self.num_nodes[g_index])
        if batch_size == -1 or batch_size > len(self.avail_node_set[g_index][s_index]):
            mask[self.avail_node_set[g_index][s_index]] = 1
            return mask
        else:
            p = self.node_weights[g_index][s_index]
            p /= np.sum(p)
            n_index = np.random.choice(
                self.node_set[g_index][s_index], size=batch_size, p=p, replace=False
            ).astype("int")
            mask[n_index] = 1
            return mask

    def sample_nodes(self, g_index, s_index, size=None, bias=1):
        mask = np.zeros(self.num_nodes[g_index])
        if "sampling_bias" in self.params:
            if self.params["sampling_bias"] > 0:
                bias /= -self.params["sampling_bias"]
            else:
                bias = 0

        if size is None or size > len(self.avail_node_set[g_index][s_index]):
            mask[self.avail_node_set[g_index][s_index]] = 1
        else:
            p = self.node_weights[g_index][s_index] ** (-bias)
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

        for n in graphs:
            self.node_weights[n] = dict()
            self.state_weights[n] = dict()

            for i, s in enumerate(inputs[n]):
                self.node_weights[n][i] = np.zeros(self.num_nodes[n])
                self.node_weights[n][i][self.avail_node_set[n][i]] = 1
                self.state_weights[n][i] = np.sum(self.node_weights[n][i])

            self.graph_weights[n] = np.sum(
                [self.state_weights[n][int(i)] for i in self.avail_state_set[n]]
            )
