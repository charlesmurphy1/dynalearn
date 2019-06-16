import numpy as np


class Sampler(object):
    def __init__(self, batch_size=None, verbose=1):

        self.verbose = verbose
        self.params = dict()
        self.params["batch_size"] = batch_size
        self.num_nodes = dict()
        self.num_samples = dict()
        self.with_weights = False

        self.graph_set = list()
        self.avail_graph_set = list()

        self.state_set = dict()
        self.avail_state_set = dict()

        self.node_set = dict()
        self.avail_node_set = dict()

    def __call__(self):

        g_index = self.get_graph()
        s_index = self.get_state(g_index)
        n_index = self.get_nodes(g_index, s_index)

        return g_index, s_index, n_index

    def update(self, graphs, inputs):
        for n in graphs:
            adj = graphs[n]
            self.num_samples[n] = inputs[n].shape[0]
            self.num_nodes[n] = inputs[n].shape[1]

            self.node_set[n] = np.arange(self.num_nodes[n]).astype("int")
            self.avail_node_set[n] = np.arange(self.num_nodes[n]).astype("int")

            self.state_set[n] = list(range(self.num_samples[n]))
            self.avail_state_set[n] = list(range(self.num_samples[n]))

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

    def sample_nodes(self, g_index, num_nodes):
        return np.random.choice(self.avail_node_set[g_index], num_nodes, replace=False)

    def get_graph(self):
        raise NotImplementedError()

    def get_state(self, g_index):
        raise NotImplementedError()

    def get_nodes(self, g_index, s_index):
        mask = np.zeros(self.num_nodes[g_index])
        if self.params["batch_size"] is None or self.params["batch_size"] > len(
            self.avail_node_set[g_index]
        ):
            mask[self.avail_node_set[g_index]] = 1
            return mask
        else:
            p = np.zeros(self.num_nodes[g_index])
            p[self.avail_node_set[g_index]] = 1
            p /= np.sum(p)
            n_index = np.random.choice(
                self.node_set[g_index],
                size=self.params["batch_size"],
                p=p,
                replace=False,
            ).astype("int")
            mask[n_index] = 1
            return mask

    def update_weights(self, graphs, inputs):
        return
