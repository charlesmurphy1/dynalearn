from .base_sampler import Sampler
import numpy as np


class RandomSampler(Sampler):
    def __init__(self, batch_size=None, replace=False, verbose=1):
        super(RandomSampler, self).__init__(batch_size, verbose)
        self.params["replace"] = replace
        self.node_weights = dict()
        self.state_weights = dict()
        self.graph_weights = dict()
        self.with_weights = True

    def get_graph(self):
        x = self.avail_graph_set
        p = np.array([self.graph_weights[xx] for xx in x])
        p /= np.sum(p)
        g_index = np.random.choice(x, p=p)
        return g_index

    def get_state(self, g_index):
        x = self.avail_state_set[g_index]
        p = np.array([self.state_weights[g_index][xx] for xx in x])
        p /= np.sum(p)
        s_index = np.random.choice(x, p=p)
        if not self.params["replace"]:
            self.avail_state_set[g_index].remove(s_index)
            self.update_set(g_index)
        return s_index

    def get_nodes(self, g_index, s_index):
        mask = np.zeros(self.num_nodes[g_index])
        if self.params["batch_size"] is None or self.params["batch_size"] > len(
            self.avail_node_set[g_index]
        ):
            mask[self.avail_node_set[g_index]] = 1
            return mask
        else:
            p = np.zeros(self.num_nodes[g_index])
            p[self.avail_node_set[g_index]] = self.node_weights[g_index][
                s_index, self.avail_node_set[g_index]
            ]
            p /= np.sum(p)
            n_index = np.random.choice(
                self.node_set[g_index],
                size=self.params["batch_size"],
                p=p,
                replace=False,
            ).astype("int")
            mask[n_index] = 1
            return mask

    def sample_nodes(self, g_index, num_nodes):
        weights = np.sum(self.node_weights[g_index], axis=0)
        p = weights[self.avail_node_set[g_index]] / np.sum(
            weights[self.avail_node_set[g_index]]
        )
        return np.random.choice(
            self.avail_node_set[g_index], num_nodes, replace=False, p=p
        )

    def update_weights(self, graphs, inputs):
        self.node_weights = dict()
        self.state_weights = dict()
        self.graph_weights = dict()
        for n in graphs:
            self.node_weights[n] = np.ones(inputs[n].shape)
            self.state_weights[n] = {
                i: np.sum(self.node_weights[n][int(i), :])
                for i in self.avail_state_set[n]
            }
            self.graph_weights[n] = np.sum(
                [self.node_weights[n][int(i), :] for i in self.avail_state_set[n]]
            )
