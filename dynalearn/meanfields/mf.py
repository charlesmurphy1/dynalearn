from .base_meanfield import BaseMeanField
import numpy as np
import networkx as nx
from scipy.special import gammaln, binom
import tqdm
from .utilities import config_k_l_grid

np.seterr(divide="ignore", invalid="ignore")


class MF(BaseMeanField):
    def __init__(self, s_dim, p_k, max_iter=100, tol=1e-3, verbose=1):
        self.s_dim = s_dim
        self.p_k = p_k
        self.k_min = self.p_k.values.min()
        self.k_max = self.p_k.values.max()
        self.k_dim = self.k_max - self.k_min + 1
        array_shape = (self.s_dim, self.k_dim)

        self.k_grid, self.l_grid = config_k_l_grid(
            p_k.values, np.arange(self.k_max + 1), s_dim
        )
        self.good_config = self.l_grid.sum(0) == self.k_grid
        self.bad_config = (self.l_grid.sum(0) > self.k_grid) + (
            self.l_grid.sum(0) < self.k_grid
        )
        self.k_grid[self.bad_config] = 0
        self.l_grid[:, self.bad_config] = 0
        super(MF, self).__init__(array_shape, max_iter, tol, verbose)

    def application(self, x):
        _x = x.reshape(self.array_shape)
        q = (
            (_x * self.p_k.values)
            @ self.p_k.weights
            / (self.p_k.values @ self.p_k.weights)
        )

        m_k = self.multinomial(q)
        new_x = np.zeros(self.array_shape)
        for i in range(self.s_dim):
            new_x += (self.ltp[i] * m_k).sum(
                tuple([-i - 1 for i in range(self.s_dim)])
            ) * _x[i]
        new_x = self.normalize_state(new_x)
        return new_x.reshape(-1)

    def multinomial(self, q):
        log_m_k = gammaln(self.k_grid + 1) + np.sum(
            (self.l_grid.T * np.log(q)).T - gammaln(self.l_grid + 1), 0
        )
        m_k = np.exp(log_m_k)
        m_k[self.bad_config] = 0
        return m_k

    def to_compartment(self, graph, state):
        x = np.zeros(self.array_shape)
        adj = nx.to_numpy_array(graph)
        for i, s in enumerate(state):
            s = s.astype("int")
            k = adj[i].sum()
            k_ind = np.where(self.p_k.values == k)[0]
            if len(k_ind) > 0:
                x[s, k_ind[0]] += 1
        x = self.normalize_state(x, clip=False)
        return x.reshape(-1)

    def to_avg(self, x):
        _x = x.reshape(self.array_shape)
        _x = _x @ self.p_k.weights
        return _x.reshape(-1)

    def normalize_state(self, x, clip=True):
        _x = x.copy()
        if clip:
            _x[_x >= 1] = 1 - 1e-15
            _x[_x <= 0] = 1e-15
        z = _x.sum(0)
        normed_x = _x / z
        return normed_x


# class LearnedModelMFMetrics(DiscreteMFMetrics):
#     def __init__(self, graph, model, max_iter=100, tol=1e-3, verbose=1):
#         self.model = model
#         self.model.num_nodes = max(dict(g.degree()).values())
#         super(LearnedModelMFMetrics, self).__init__(
#             graph, self.model.num_states, self.model_ltp, max_iter, tol, verbose
#         )
#
#     def model_ltp(self, k):
#         state_degree = np.array(all_combinations(k, self.num_states))
#         adj = np.zeros((self.model.num_nodes, self.model.num_nodes))
#         adj[1 : k + 1, 0] = 1
#         adj[0, 1 : k + 1] = 1
#         val = np.zeros((len(state_degree), self.num_states, self.num_states))
#         for i, s in enumerate(state_degree):
#             neighbors_states = np.concatenate([j * np.ones(l) for j, l in enumerate(s)])
#             inputs = np.zeros(k + 1)
#             inputs[1:] = neighbors_states
#             for j in range(self.num_states):
#                 inputs[0] = j
#                 p = self.model.predict(inputs, adj)[0]
#                 val[i, j, :] = p
#         return val
