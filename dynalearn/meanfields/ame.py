from .base_meanfield import BaseMeanField
from itertools import product
import networkx as nx
import numpy as np
import tqdm
from .utilities import config_k_l_grid


class AME(BaseMeanField):
    def __init__(self, s_dim, p_k, max_iter=100, tol=1e-3, verbose=1):
        self.s_dim = s_dim
        self.p_k = p_k
        self.k_min = self.p_k.values.min()
        self.k_max = self.p_k.values.max()
        self.k_dim = self.k_max - self.k_min + 1
        array_shape = (self.s_dim, self.k_dim, *[self.k_max + 1] * self.s_dim)

        self.k_grid, self.l_grid = config_k_l_grid(
            p_k.values, np.arange(self.k_max + 1), s_dim
        )
        self.k_grid = self.k_grid.astype("int")
        self.l_grid = self.l_grid.astype("int")

        self.good_config = self.l_grid.sum(0) == self.k_grid
        self.bad_config = (self.l_grid.sum(0) > self.k_grid) + (
            self.l_grid.sum(0) < self.k_grid
        )
        super(AME, self).__init__(array_shape, max_iter, tol, verbose)

    def application(self, x):
        _x = x.reshape(self.array_shape)

        # Computing q fields (mu, xi, xi')
        q = self.__app_q(_x)

        # Computing first term (mu, k, l), mu is fixed here
        first_term = np.zeros(self.array_shape)
        for i in range(self.s_dim):
            first_term += self.ltp[i] * _x[i]

        second_term = np.zeros(self.array_shape)
        for i, j in product(range(self.s_dim), range(self.s_dim)):
            if i == j:
                continue
            second_term -= (_x.T * q[:, i, j]).T * self.l_grid[i]

        third_term = np.zeros(self.array_shape)
        for i, j in product(range(self.s_dim), range(self.s_dim)):
            y = _x.copy()
            y = np.roll(y, 1, 2 + i)
            y = np.roll(y, -1, 2 + j)
            y[:, self.l_grid[i] == 0] = 0
            if i == j:
                continue
            third_term += (y.T * q[:, j, i]).T * (self.l_grid[j] + 1)

        new_x = first_term + second_term + third_term

        # normalizing new_x
        new_x = self.normalize_state(new_x)

        return new_x.reshape(-1)

    def __app_q(self, x):
        # Computing numerator (mu, xi, xi')
        x1 = np.zeros([self.s_dim] * 3)
        for i in range(self.s_dim):
            l = self.l_grid[i]
            term = np.zeros(self.ltp.shape)
            for j, k in product(range(self.s_dim), range(self.s_dim)):
                term[j, k] = l * x[j] * self.ltp[j, k]
            for j in range(self.s_dim):
                term = term.sum(-1)
            x1[i] = term @ self.p_k.weights

        # Computing numerator (mu, xi)
        x2 = np.zeros([self.s_dim] * 2)
        for i in range(self.s_dim):
            l = self.l_grid[i]
            term = x * l
            for j in range(self.s_dim):
                term = term.sum(-1)
            x2[i] = term @ self.p_k.weights
        q = (x1.T / x2.T).T

        # normalizing q
        z = q.sum(-1)
        q = (q.T / z.T).T
        return q

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
        x = self.normalize_state(x, clip=False)
        return x.reshape(-1)

    def to_avg(self, x):
        _x = x.reshape(self.array_shape)
        y = _x.sum(-1)
        for i in range(self.s_dim - 1):
            y = y.sum(-1)
        return y @ self.p_k.weights

    def normalize_state(self, x, clip=True):
        _x = x.copy()
        if clip:
            _x[_x >= 1] = 1 - 1e-15
            _x[_x <= 0] = 1e-15
            _x[:, self.bad_config] = 0
        z = _x.sum(0)
        for i in range(self.s_dim):
            z = z.sum(-1)
        normed_x = _x.copy()
        for i in range(self.s_dim):
            normed_x[i] = (_x[i].T / z).T
        return normed_x
