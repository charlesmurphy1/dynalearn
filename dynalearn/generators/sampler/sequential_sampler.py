from .base_sampler import Sampler
import numpy as np


class SequentialSampler(Sampler):
    def __init__(self, verbose=1):
        super(SequentialSampler, self).__init__(verbose)

    def get_graph(self):
        g_index = self.graph_indexes[0]
        return g_index

    def get_state(self, g_index):
        s_index = self.state_indexes[g_index][0]
        self.update_indexes(g_index, s_index)
        return s_index

    def get_nodes(self, g_index, s_index):
        w = np.zeros(self.num_nodes[g_index])
        w[self.avail_nodeset[g_index]] = 1
        return w
