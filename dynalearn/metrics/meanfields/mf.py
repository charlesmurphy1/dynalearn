from .base_meanfield import BaseMeanField
import numpy as np
import networkx as nx
from scipy.special import gammaln, binom
import tqdm
from .utilities import config_k_l_grid

np.seterr(divide="ignore", invalid="ignore")


class MF(BaseMeanField):
    def __init__(self, s_dim, degree_dist, tol=1e-5, verbose=0, dtype="float"):
        self.s_dim = s_dim
        self.degree_dist = degree_dist
        super(MF, self).__init__(self.array_shape, tol=tol, verbose=verbose)

    def application(self, x):
        _x = x.reshape(self.array_shape)
        _phi = self.app_phi(_x)
        new_x = self.app_x(_x, _phi)
        new_x = self.clip(new_x)
        new_x = self.normalize_state(new_x)
        return new_x.reshape(-1)

    def app_x(self, x, phi):
        m_k = self.multinomial(phi)
        new_x = np.zeros(self.array_shape)
        for i in range(self.s_dim):
            new_x += (self.ltp[i] * m_k).sum(
                tuple([-i - 1 for i in range(self.s_dim)])
            ) * x[i]
        return new_x

    def app_phi(self, x):
        new_phi = (
            (x * self.degree_dist.values)
            @ self.degree_dist.weights
            / (self.degree_dist.values @ self.degree_dist.weights)
        )
        new_phi = self.clip(new_phi)
        new_phi /= new_phi.sum()
        return new_phi

    def multinomial(self, phi):
        log_m_k = gammaln(self.k_grid + 1) + np.sum(
            (self.l_grid.T * np.log(phi)).T - gammaln(self.l_grid + 1), 0
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
            k_ind = np.where(self.degree_dist.values == k)[0]
            if len(k_ind) > 0:
                x[s, k_ind[0]] += 1
        x = self.normalize_state(x)
        return x.reshape(-1)

    def to_avg(self, x):
        _x = x.reshape(self.array_shape)
        _x = _x @ self.degree_dist.weights
        return _x.reshape(-1)

    def normalize_state(self, x):
        _x = x.copy()
        z = _x.sum(0)
        normed_x = _x / z
        return normed_x

    @property
    def degree_dist(self):
        if self._degree_dist is None:
            raise NotImplementedError(
                "No degree distribution has been given to the meanfield."
            )
        return self._degree_dist

    @degree_dist.setter
    def degree_dist(self, degree_dist):
        self._degree_dist = degree_dist
        self.k_min = self.degree_dist.values.min()
        self.k_max = self.degree_dist.values.max()
        self.k_dim = int(self.k_max - self.k_min + 1)
        self.array_shape = (self.s_dim, self.k_dim)

        self.k_grid, self.l_grid = config_k_l_grid(
            degree_dist.values, np.arange(self.k_max + 1), self.s_dim
        )
        self.good_config = self.l_grid.sum(0) == self.k_grid
        self.bad_config = (self.l_grid.sum(0) > self.k_grid) + (
            self.l_grid.sum(0) < self.k_grid
        )
        self.k_grid[self.bad_config] = 0
        self.l_grid[:, self.bad_config] = 0
        self.ltp = self.compute_ltp()
