from .base_sampler import Sampler
import numpy as np


class RandomSampler(Sampler):
    def __init__(self, batchsize=None, sample_from_weight=1, verbose=1, replace=0):
        super(RandomSampler, self).__init__(verbose)
        self.params["batchsize"] = batchsize
        self.params["sample_from_weight"] = sample_from_weight
        self.params["replace"] = replace

    def get_graph(self):
        g_index = np.random.choice(self.graph_indexes)
        return g_index

    def get_state(self, g_index):
        s_index = np.random.choice(self.state_indexes[g_index])
        if self.params["replace"]:
            self.update_indexes(g_index, s_index)
        return s_index

    def get_nodes(self, g_index, s_index):
        w = np.zeros(self.num_nodes[g_index])
        if self.params["batchsize"] is None or self.params["batchsize"] > len(
            self.avail_nodeset[g_index]
        ):
            w[self.avail_nodeset[g_index]] = 1
            return w
        else:
            w[self.avail_nodeset[g_index]] = self.params["batchsize"] / len(
                self.avail_nodeset[g_index]
            )
            if self.params["sample_from_weight"]:
                w[w > 1] = 1
                return np.random.binomial(1, w)
            else:
                return w
