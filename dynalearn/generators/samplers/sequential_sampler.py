from .base_sampler import Sampler
import numpy as np


class SequentialSampler(Sampler):
    def __init__(self, name, verbose=1, sample_from_weights=True, resample=-1):
        super(SequentialSampler, self).__init__(
            name, verbose, sample_from_weights, resample
        )

    def get_graph(self):
        g_index = self.avail_graph_set[0]
        return g_index

    def get_state(self, g_index):
        s_index = self.avail_state_set[g_index][0]
        self.avail_state_set[g_index].remove(s_index)
        self.update_set(g_index)
        return s_index
