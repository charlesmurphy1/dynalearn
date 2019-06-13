import numpy as np


class Sampler(object):
    def __init__(self, verbose=1):

        self.verbose = verbose
        self.graph_indexes = list()
        self.state_indexes = dict()
        self.sample_size = 0
        self.num_nodes = dict()
        self.avail_nodeset = dict()
        self.params = dict()
        self.with_weights = False

    def __call__(self):

        g_index = self.get_graph()
        s_index = self.get_state(g_index)
        n_index = self.get_nodes(g_index, s_index)

        return g_index, s_index, n_index

    def update(self, graphs, inputs, targets, val_size=None, test_size=None):
        for n in graphs:
            adj = graphs[n]
            num_samples = inputs[n].shape[0]
            self.num_nodes[n] = inputs[n].shape[1]
            self.avail_nodeset[n] = np.arange(self.num_nodes[n]).astype("int")
            self.sample_size += num_samples
            self.state_indexes[n] = list(range(num_samples))
        self.graph_indexes = list(graphs.keys())
        self.update_weights(graphs, inputs, targets)

        return

    def reset_indexes(self):
        self.graph_indexes = list(self.graphs.keys())
        for g_index in self.graph_indexes:
            num_sample = self.inputs[g_index].shape[0]
            self.state_indexes[g_index] = list(range(num_sample))
        return

    def update_indexes(self, g_index, index):
        if index in self.state_indexes[g_index]:
            self.state_indexes[g_index].remove(index)
        if len(self.state_indexes[g_index]) == 0:
            if g_index in self.graph_indexes:
                self.graph_indexes.remove(g_index)

        if len(self.graph_indexes) == 0:
            self.reset_indexes()
        return

    def val(self):
        self.validation_mode = True
        self.test_mode = False

    def test(self):
        self.validation_mode = False
        self.test_mode = True

    def train(self):
        self.validation_mode = False
        self.test_mode = False

    def sample_nodes(self, g_index, num_nodes):
        return np.random.choice(self.avail_nodeset[g_index], num_nodes, replace=False)

    def get_graph(self):
        raise NotImplementedError()

    def get_state(self, g_index):
        raise NotImplementedError()

    def get_nodes(self, g_index, s_index):
        raise NotImplementedError()

    def update_weights(self, graphs, inputs, targets):
        return
