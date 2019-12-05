from .mf import MF
from .utilities import *
import numpy as np
import tqdm
from scipy.special import binom


def all_combinations(k, d):
    if d == 1:
        return [[k]]
    return [(*j, k - i) for i in range(k + 1) for j in all_combinations(i, d - 1)]


class GNN_MF(MF):
    def __init__(self, model, degree_dist, tol=1e-3, verbose=1):
        self.model = model
        super(GNN_MF, self).__init__(
            self.model.num_states, degree_dist, tol=tol, verbose=verbose
        )

    def compute_ltp(self,):
        self.model.num_nodes = self.k_max + 1

        ltp = np.zeros((self.s_dim, self.s_dim, *self.k_grid.shape))
        for k_ind, k in enumerate(self.degree_dist.values):
            state_degree = np.array(all_combinations(k, self.s_dim))
            adj = np.zeros((self.model.num_nodes, self.model.num_nodes))
            adj[1 : k + 1, 0] = 1
            adj[0, 1 : k + 1] = 1
            for s in state_degree:
                neighbors_states = np.concatenate(
                    [ss * np.ones(l) for ss, l in enumerate(s)]
                )
                if neighbors_states.shape[0] < self.model.num_nodes:
                    neighbors_states = np.concatenate(
                        (
                            neighbors_states,
                            np.zeros(
                                self.model.num_nodes - neighbors_states.shape[0] - 1
                            ),
                        )
                    )

                inputs = np.zeros(self.model.num_nodes)
                inputs[1:] = neighbors_states
                for i in range(self.s_dim):
                    inputs[0] = i
                    p = self.model.predict(inputs, adj)[0]
                    for j in range(self.s_dim):
                        ltp[(i, j, k_ind, *s)] = p[j]
        return ltp
