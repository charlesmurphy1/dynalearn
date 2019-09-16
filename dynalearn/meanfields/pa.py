from .base_meanfield import BaseMeanField
from itertools import product
import networkx as nx
import numpy as np
from scipy.special import gammaln, binom
import tqdm
from .utilities import config_k_l_grid

np.seterr(divide="ignore", invalid="ignore")


class PA(BaseMeanField):
    def __init__(self, s_dim, p_k, max_iter=100, tol=1e-3, verbose=1):
        self.s_dim = s_dim
        self.p_k = p_k
        self.k_min = self.p_k.values.min()
        self.k_max = self.p_k.values.max()
        self.k_dim = self.k_max - self.k_min + 1
        array_shape = (self.s_dim + self.s_dim ** 2, self.k_dim)

        self.k_grid, self.l_grid = config_k_l_grid(
            p_k.values, np.arange(self.k_max + 1), s_dim
        )
        self.good_config = self.l_grid.sum(0) == self.k_grid
        self.bad_config = (self.l_grid.sum(0) > self.k_grid) + (
            self.l_grid.sum(0) < self.k_grid
        )
        # self.k_grid[self.bad_config] = 0
        # self.l_grid[:, self.bad_config] = 0
        super(PA, self).__init__(array_shape, max_iter, tol, verbose)

    def random_state(self):
        _x = self.__random_state()
        _phi = self.__random_phi_field()
        return self.__group_state(_x, _phi).reshape(-1)

    def abs_state(self, s):
        _x = np.zeros(self.s_dim, self.k_dim)
        _x[s] = 1
        _phi = np.zeros(self.s_dim, self.s_dim, self.k_dim)
        _phi[:, s, :] = 1
        return self.__group_state(_x, _phi).reshape(-1)

    def application(self, x):
        x = x.reshape(self.array_shape)
        _x, _phi = self.__separate_state(x)

        m_k = self.multinomial(_phi)
        __phi = self.__app_phi(_x, _phi)
        __x = np.zeros(_x.shape)
        for i, j in product(range(self.s_dim), range(self.s_dim)):
            __x[i] += _x[j] * (self.ltp[j, i] * m_k[j]).sum(
                tuple([-k - 1 for k in range(self.s_dim)])
            )
        new_x = self.__group_state(__x, __phi)
        new_x = self.normalize_state(new_x)
        return new_x.reshape(-1)

    def __app_phi(self, x, phi):
        m_k = self.multinomial(phi)
        num = np.zeros((self.s_dim, *x.shape))
        den = np.zeros((self.s_dim, *x.shape))
        for i, j, n in product(range(self.s_dim), range(self.s_dim), range(self.s_dim)):
            num[i, j] += x[n] * (self.ltp[n, i] * m_k[n] * self.l_grid[j]).sum(
                tuple([-k - 1 for k in range(self.s_dim)])
            )
            den[i, j] += x[n] * (self.ltp[n, i] * m_k[n] * self.k_grid).sum(
                tuple([-k - 1 for k in range(self.s_dim)])
            )

        new_phi = num / den

        return new_phi

    def multinomial(self, phi):
        log_m_k = np.zeros((self.s_dim, *self.k_grid.shape))
        for i, j in product(range(self.s_dim), range(self.s_dim)):
            log_m_k[i] += self.l_grid[j] * np.log(phi[i, j]) - gammaln(
                self.l_grid[j] + 1
            )
        log_m_k += gammaln(self.k_grid + 1)
        m_k = np.exp(log_m_k)
        m_k[:, self.bad_config] = 0

        return m_k

    def to_compartment(self, graph, state):
        x = np.zeros(self.array_shape)
        _x, _phi = self.__separate_state(x)
        adj = nx.to_numpy_array(graph)
        for i, s in enumerate(state):
            s = s.astype("int")
            k = adj[i].sum()
            k_ind = np.where(self.p_k.values == k)[0]
            if len(k_ind) > 0:
                _x[s, k_ind[0]] += 1
        _x = self.__normalize_state(_x, clip=False)
        _phi = self.__get_phi_field(graph, state)
        x = self.__group_state(_x, _phi)
        return x.reshape(-1)

    def to_avg(self, x):
        x = x.reshape(self.array_shape)
        _x, _phi = self.__separate_state(x)
        _x = _x @ self.p_k.weights
        return _x.reshape(-1)

    def normalize_state(self, x, clip=True):
        _x = x.copy()
        # if clip:
        #     _x[_x >= 1] = 1 - 1e-15
        #     _x[_x <= 0] = 1e-15
        _x, _phi = self.__separate_state(_x)
        _x = self.__normalize_state(_x)
        _phi = self.__normalize_phi_field(_phi)
        normed_x = self.__group_state(_x, _phi)
        return normed_x

    def __get_phi_field(self, graph, state):
        adj = nx.to_numpy_array(graph)
        _phi = np.zeros((self.s_dim, self.s_dim, self.k_dim))
        for i, s in enumerate(state):
            k = adj[i].sum()
            k_ind = np.where(self.p_k.values == k)[0]
            for j, _s in enumerate(state[adj[i] == 1]):
                if len(k_ind) > 0:
                    _phi[int(s), int(_s), int(k_ind[0])] += 1
        _phi = self.__normalize_phi_field(_phi)
        return _phi

    def __normalize_state(self, x, clip=True):
        z = x.sum(0)
        return x / z

    def __normalize_phi_field(self, phi, clip=True):
        z = phi.sum(1)
        for i in range(self.s_dim):
            phi[i] /= z[i]
        return phi

    def __separate_state(self, x):
        _x = x[: self.s_dim]
        _phi = x[self.s_dim :].reshape(self.s_dim, self.s_dim, -1)
        return _x, _phi

    def __group_state(self, x, phi):
        grouped_x = np.zeros(self.array_shape)
        grouped_x[: self.s_dim] = x
        grouped_x[self.s_dim :] = phi.reshape(self.s_dim * self.s_dim, -1)
        return grouped_x

    def __random_state(self):
        x = np.random.rand(self.s_dim, self.k_dim)
        return self.__normalize_state(x)

    def __random_phi_field(self):
        phi = np.random.rand(self.s_dim, self.s_dim, self.k_dim)
        return self.__normalize_phi_field(phi)
