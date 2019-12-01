import numpy as np


class Sampler(object):
    def __init__(self, name, verbose=1, sample_from_weights=True, resample=-1):
        self.name = name
        self.verbose = verbose
        self.sample_from_weights = sample_from_weights
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
        if self.sample_from_weights:
            if batch_size == -1 or batch_size > len(
                self.avail_node_set[g_index][s_index]
            ):
                mask[self.avail_node_set[g_index][s_index]] = 1
                return mask
            else:
                p = self.node_weights[g_index][s_index]
                p /= np.sum(p)
                n_index = np.random.choice(
                    self.node_set[g_index][s_index], size=batch_size, p=p, replace=False
                ).astype("int")
                mask[n_index] = 1
        else:
            if batch_size == -1 or batch_size > len(
                self.avail_node_set[g_index][s_index]
            ):
                p = self.node_weights[g_index][s_index]
                p /= np.sum(p)
                p = p[self.avail_node_set[g_index][s_index]]
                mask[self.avail_node_set[g_index][s_index]] = p * np.sum(p > 0)
                return mask
            else:
                p = self.node_weights[g_index][s_index]
                p /= np.sum(p)
                n_index = np.random.choice(
                    self.node_set[g_index][s_index], size=batch_size, p=p, replace=False
                ).astype("int")
                p = p[n_index]
                mask[n_index] = p * np.sum(p > 0)
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

    def save(self, h5file, overwrite=False):
        if "sampler/" + self.name in h5file:
            if overwrite:
                del h5file["sampler/" + self.name]
            else:
                return
        h5file.create_group("sampler/" + self.name)
        h5group = h5file["sampler/" + self.name]
        for g_name in self.graph_set:
            num_samples = self.node_weights[g_name][0].shape[0]
            avail_node_set = np.array(
                [
                    np.where(
                        self.node_weights[g_name][i] > 0,
                        np.ones(self.node_weights[g_name][i].shape),
                        np.zeros(self.node_weights[g_name][i].shape),
                    )
                    for i in range(inputs.shape[0])
                ]
            )
            node_weights = np.array(
                [self.node_weights[g_name][i] for i in range(inputs.shape[0])]
            )
            state_weights = np.array(
                [self.state_weights[g_name][i] for i in range(inputs.shape[0])]
            )
            graph_weights = self.graph_weights[g_name]
            h5group.create_dataset(g_name + "/avail_node_set", data=avail_node_set)
            h5group.create_dataset(g_name + "/node_weights", data=node_weights)
            h5group.create_dataset(g_name + "/state_weights", data=state_weights)
            h5group.create_dataset(g_name + "/graph_weights", data=graph_weights)

    def load_sampler(self, h5file):
        if "sampler/" + self.name in h5file:
            for k, v in h5file["sampler/" + self.name].items():
                avail_node_set = v["avail_node_set"][...]
                node_weights = v["node_weights"][...]
                state_weights = v["state_weights"][...]
                graph_weights = v["graph_weights"][...]
                self.avail_node_set[k] = {
                    i: np.argwhere(nodes) for i, nodes in enumerate(avail_node_set)
                }
                self.node_weights[k] = {
                    i: weights for i, weights in enumerate(node_weights)
                }
                self.state_weights[k] = {
                    i: weights for i, weights in enumerate(state_weights)
                }
                self.graph_weights[k] = graph_weights
