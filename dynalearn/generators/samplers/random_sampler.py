from .base_sampler import Sampler
import numpy as np


class RandomSampler(Sampler):
    def __init__(
        self, name, replace=False, verbose=0, sample_from_weights=True, resample=-1
    ):
        super(RandomSampler, self).__init__(
            name, verbose, sample_from_weights, resample
        )
        self.replace = replace

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
        if not self.replace:
            self.avail_state_set[g_index].remove(s_index)
            self.update_set(g_index)
        return s_index
