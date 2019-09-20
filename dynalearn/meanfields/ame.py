from .base_meanfield import BaseMeanField
from itertools import product
import networkx as nx
import numpy as np
import tqdm
from .utilities import config_k_l_grid


class AME(BaseMeanField):
    def __init__(self, s_dim, p_k, tol=1e-3, verbose=1):
        self.s_dim = s_dim
        self.p_k = p_k
        self.k_min = self.p_k.values.min()
        self.k_max = self.p_k.values.max()
        self.k_dim = self.k_max - self.k_min + 1

        self.k_grid, self.l_grid = config_k_l_grid(
            p_k.values, np.arange(self.k_max + 1), s_dim
        )
        self.k_grid = self.k_grid.astype("int")
        self.l_grid = self.l_grid.astype("int")

        self.good_config = self.l_grid.sum(0) == self.k_grid
        self.l_dim = self.good_config.sum()
        self.bad_config = (self.l_grid.sum(0) > self.k_grid) + (
            self.l_grid.sum(0) < self.k_grid
        )
        array_shape = (self.s_dim, self.k_dim, *[self.k_max + 1] * self.s_dim)
        super(AME, self).__init__(array_shape, tol, verbose)

    def application(self, x):
        _x = x.reshape(self.array_shape)

        _q = self.app_q(_x)
        new_x = self.app_x(_x, _q)
        new_x = self.normalize_state(new_x)

        return new_x.reshape(-1)

    def app_x(self, x, q):
        # Computing first term (mu, k, l), mu is fixed here
        _ltp = self.ltp[:, :, self.good_config]
        _x = x[:, self.good_config]
        _l = self.l_grid[:, self.good_config]
        _k = self.k_grid[self.good_config]
        first_term = np.zeros(_x.shape)
        for i in range(self.s_dim):
            first_term += _ltp[i] * _x[i]

        second_term = np.zeros(_x.shape)
        for i, j in product(range(self.s_dim), range(self.s_dim)):
            if i == j:
                continue
            second_term += (_x.T * q[:, i, j]).T * _l[i]

        third_term = np.zeros(_x.shape)
        for i, j in product(range(self.s_dim), range(self.s_dim)):
            if i == j:
                continue
            term = (_x.T * q[:, j, i]).T * _l[j]
            _term = np.zeros(self.array_shape)
            _term[:, self.good_config] = term
            _term = np.roll(_term, 1, 2 + i)
            _term = np.roll(_term, -1, 2 + j)
            _term[:, self.l_grid[i] == 0] = 0
            third_term += _term[:, self.good_config]

        new_x = first_term - second_term + third_term
        _new_x = np.zeros(self.array_shape)
        _new_x[:, self.good_config] = new_x
        return _new_x

    def app_q(self, x):
        num = np.zeros((self.s_dim, self.s_dim, self.s_dim))
        den = np.zeros((self.s_dim, self.s_dim, self.s_dim))
        for i, j, k in product(range(self.s_dim), range(self.s_dim), range(self.s_dim)):
            num[i, j, k] = (
                self.__marginalize_on_l(self.l_grid[i] * x[j] * self.ltp[j, k])
                @ self.p_k.weights
            )
            den[i, j, k] = (
                self.__marginalize_on_l(self.l_grid[i] * x[j]) @ self.p_k.weights
            )

        new_q = num / den
        return new_q

    def to_compartment(self, graph, state):
        x = np.zeros(self.array_shape)
        adj = nx.to_numpy_array(graph)
        for i, s in enumerate(state):
            s = s.astype("int")
            l = np.array([adj[i] @ (state == j) for j in range(self.s_dim)]).astype(
                "int"
            )
            k_ind = np.where(self.p_k.values == l.sum())[0]
            if len(k_ind) > 0:
                x[(s, k_ind[0], *l)] += 1
        x = self.normalize_state(x, overclip=False)
        return x.reshape(-1)

    def to_avg(self, x):
        _x = x.reshape(self.array_shape)
        y = _x.sum(-1)
        for i in range(self.s_dim - 1):
            y = y.sum(-1)
        return y @ self.p_k.weights

    def normalize_state(self, x, overclip=True):
        _x = x.copy()
        _x[_x <= 0] = 1e-15
        _x[:, self.bad_config] = 0
        if overclip:
            _x[_x >= 1] = 1 - 1e-15
        z = _x.sum(0)
        for i in range(self.s_dim):
            z = z.sum(-1)
        normed_x = _x.copy()
        for i in range(self.s_dim):
            normed_x[i] = (_x[i].T / z).T
        return normed_x

    def __marginalize_on_l(self, x):
        _x = x.copy()
        _x.T[self.bad_config.T] = 0
        for i in range(self.s_dim):
            _x = _x.sum(-1)
        return _x

    def abs_state(self, s):
        x = np.zeros(self.array_shape)
        ind = self.l_grid[s] == self.k_grid
        x[s, ind] = 1
        return self.normalize_state(x).reshape(-1)
