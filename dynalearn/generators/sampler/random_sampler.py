from .base_sampler import Sampler
import numpy as np


class RandomSampler(Sampler):
    def __init__(self, replace=False, verbose=1):
        super(RandomSampler, self).__init__(verbose)
        self.params["replace"] = replace

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

    # def get_nodes(self, g_index, s_index, batch_size=None):
    #     mask = np.zeros(self.num_nodes[g_index])
    #     # print(self.avail_node_set[g_index].keys(), s_index)
    #     if (
    #         batch_size is None
    #         or batch_size > self.avail_node_set[g_index][s_index].shape[0]
    #     ):
    #         mask[self.avail_node_set[g_index][s_index]] = 1
    #         return mask
    #     else:
    #         p = self.node_weights[g_index][s_index]
    #         p /= np.sum(p)
    #         n_index = np.random.choice(
    #             self.node_set[g_index][s_index], size=batch_size, p=p, replace=False
    #         ).astype("int")
    #         mask[n_index] = 1
    #         return mask
