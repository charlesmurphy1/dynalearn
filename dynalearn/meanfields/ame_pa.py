from .base_meanfield import BaseMeanField
from itertools import product
import networkx as nx
import numpy as np
from scipy.special import gammaln, gamma, binom
import tqdm
from .utilities import config_k_l_grid

np.seterr(divide="ignore", invalid="ignore")


class AME_PA(BaseMeanField):
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
        self.k_grid[self.bad_config] = 0
        self.l_grid[:, self.bad_config] = 0
        super(AME_PA, self).__init__(array_shape, max_iter, tol, verbose)

    def random_state(self):
        _x = self.__random_state()
        _phi = self.__random_phi_field()
        return self.__group_state(_x, _phi).reshape(-1)

    def abs_state(self, s):
        _x = np.zeros(self.s_dim, self.k_dim)
        _x[s] = 1
        _phi = self.__get_phifield(_x)
        return self.__group_state(_x, _phi).reshape(-1)

    def application(self, x):
        x = x.reshape(self.array_shape)
        _x, _phi = self.__separate_state(x)
        m_k = self.multinomial(_phi)
        __phi = self.__app_phi(_x, _phi)

        _x = (m_k.T * _x.T).T

        # Computing q fields (mu, xi, xi')
        q = self.__app_q(_x)

        # Computing first term (mu, k, l), mu is fixed here
        first_term = np.zeros((self.s_dim, *self.k_grid.shape))
        for i in range(self.s_dim):
            first_term += self.ltp[i] * _x[i]

        second_term = np.zeros((self.s_dim, *self.k_grid.shape))
        for i, j in product(range(self.s_dim), range(self.s_dim)):
            if i == j:
                continue
            second_term -= (_x.T * q[:, i, j]).T * self.l_grid[i]

        third_term = np.zeros((self.s_dim, *self.k_grid.shape))
        for i, j in product(range(self.s_dim), range(self.s_dim)):
            y = _x.copy()
            y = np.roll(y, 1, 2 + i)
            y = np.roll(y, -1, 2 + j)
            y[:, self.l_grid[i] == 0] = 0
            if i == j:
                continue
            third_term += (y.T * q[:, j, i]).T * (self.l_grid[j] + 1)

        __x = first_term + second_term + third_term
        __x = __x.sum(tuple([-k - 1 for k in range(self.s_dim)]))

        # normalizing new_x
        new_x = self.__group_state(__x, __phi)
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
            log_m_k[i] += (self.l_grid[j].T * np.log(phi[i, j])).T - gammaln(
                self.l_grid[j] + 1
            )
        m_k = np.exp(log_m_k) * gamma(self.k_grid + 1)
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
        _phi = self.__get_phifield(_x)
        x = self.__group_state(_x, _phi)
        return x.reshape(-1)

    def to_avg(self, x):
        x = x.reshape(self.array_shape)
        _x, _phi = self.__separate_state(x)
        _x = _x @ self.p_k.weights
        return _x.reshape(-1)

    def normalize_state(self, x, clip=True):
        _x = x.copy()
        if clip:
            _x[_x >= 1] = 1 - 1e-15
            _x[_x <= 0] = 1e-15
        _x, _phi = self.__separate_state(_x)
        _x = self.__normalize_state(_x)
        _phi = self.__normalize_phi_field(_phi)
        normed_x = self.__group_state(_x, _phi)
        return normed_x

    def __get_phifield(self, x):
        _phi = self.__random_phi_field().reshape(self.s_dim, self.s_dim, -1)
        diff = np.inf
        it = 0
        while diff > self.tol:
            new_phi = self.__app_phi(x, _phi)
            diff = np.abs(new_phi - _phi).mean()
            _phi = new_phi
            it += 1
            if it > self.max_iter:
                print("Not much progress is being made.")
                break
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
