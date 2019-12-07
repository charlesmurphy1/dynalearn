from .base_meanfield import BaseMeanField
from itertools import product
import networkx as nx
import numpy as np
from scipy.special import gammaln, gamma, binom
import tqdm
from .utilities import config_k_l_grid, EPSILON

np.seterr(divide="ignore", invalid="ignore")


class PA(BaseMeanField):
    def __init__(self, s_dim, p_k, params, tol=1e-3, verbose=1, dtype="float"):
        self.s_dim = s_dim
        self.p_k = p_k
        self.params = params
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

        super(PA, self).__init__(array_shape, tol=tol, verbose=verbose)

    def random_state(self):
        _x = self.__random_state()
        _phi = self.__random_phi()
        return self.group_state(_x, _phi).reshape(-1)

    def abs_state(self, s):
        _x = np.zeros((self.s_dim, self.k_dim))
        _x[s] = 1
        _x = self.__normalize_x(_x)
        _phi = self.get_phi(_x)
        _phi = self.__normalize_phi(_phi)
        # _phi = np.ones((self.s_dim, self.s_dim, self.k_dim)) / self.s_dim
        # _phi[s, :, :] = 0
        # _phi[s, s, :] = 1
        x = self.group_state(_x, _phi)
        return self.normalize_state(x).reshape(-1)

    def application(self, x):
        x = x.reshape(self.array_shape)
        _x, _phi = self.separate_state(x)
        __x = self.app_x(_x, _phi)
        __phi = self.app_phi(_x, _phi)

        # normalizing new_x
        new_x = self.group_state(__x, __phi)
        new_x = self.normalize_state(new_x)
        return new_x.reshape(-1)

    def app_x(self, x, phi):
        m_k = self.multinomial(phi)
        new_x = np.zeros(x.shape)
        for i in range(self.s_dim):
            new_x[i] = (x * self.__marginalize_on_l(self.ltp[:, i] * m_k)).sum(0)
        return new_x

    def __app_large_x(self, x, phi):
        # Computing first term (mu, k, l), mu is fixed here
        m_k = self.multinomial(phi)
        large_x = (m_k.T * x.T).T

        q = self.app_q(x, phi)
        _ltp = self.ltp[:, :, self.good_config]
        _x = large_x[:, self.good_config]
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
            _term = np.zeros(large_x.shape)
            _term[:, self.good_config] = term
            _term = np.roll(_term, 1, 2 + i)
            _term = np.roll(_term, -1, 2 + j)
            _term[:, self.l_grid[i] == 0] = 0
            third_term += _term[:, self.good_config]

        new_x = first_term - second_term + third_term
        _new_x = np.zeros(large_x.shape)
        _new_x[:, self.good_config] = new_x
        return _new_x

    def app_phi(self, x, phi):
        num = np.zeros((self.s_dim, self.s_dim, self.k_dim))
        den = np.zeros((self.s_dim, self.s_dim, self.k_dim))
        m_k = self.multinomial(phi)
        new_large_x = self.__app_large_x(x, phi)
        new_small_x = self.__marginalize_on_l(new_large_x)

        for i, j in product(range(self.s_dim), range(self.s_dim)):
            num[i, j] = self.__marginalize_on_l(self.l_grid[j] * new_large_x[i])
            den[i, j] = new_small_x[i] * self.p_k.values

        new_phi = num / den
        return new_phi

    def app_q(self, x, phi):
        num = np.zeros((self.s_dim, self.s_dim, self.s_dim))
        den = np.zeros((self.s_dim, self.s_dim, self.s_dim))
        m_k = self.multinomial(phi)
        _x = (m_k.T * x.T).T
        for i, j, k in product(range(self.s_dim), range(self.s_dim), range(self.s_dim)):
            num[i, j, k] = (
                x[j] * self.__marginalize_on_l(self.l_grid[i] * m_k[j] * self.ltp[j, k])
            ) @ self.p_k.weights
            den[i, j, k] = (x[j] * phi[j, i] * self.p_k.values) @ self.p_k.weights

        new_q = num / den
        return new_q

    def multinomial(self, phi):
        log_m_k = np.zeros((self.s_dim, *self.k_grid.shape))
        for i, j in product(range(self.s_dim), range(self.s_dim)):
            log_m_k[i] += (self.l_grid[j].T * np.log(phi[i, j])).T - gammaln(
                self.l_grid[j] + 1
            )
        log_m_k += gammaln(self.k_grid + 1)
        m_k = np.exp(log_m_k)
        m_k[:, self.bad_config] = 0
        return m_k

    def to_compartment(self, graph, state):
        x = np.zeros(self.array_shape)
        _x, _phi = self.separate_state(x)
        adj = nx.to_numpy_array(graph)
        for i, s in enumerate(state):
            s = s.astype("int")
            k = adj[i].sum()
            k_ind = np.where(self.p_k.values == k)[0]
            if len(k_ind) > 0:
                _x[s, k_ind[0]] += 1

            for j, _s in enumerate(state[adj[i] == 1]):
                if len(k_ind) > 0:
                    _phi[int(s), int(_s), int(k_ind[0])] += 1
        _x = self.__normalize_x(_x, overclip=False)
        _phi = self.__normalize_phi(_phi, overclip=False)
        x = self.group_state(_x, _phi)
        x = self.normalize_state(x)
        return x.reshape(-1)

    def to_avg(self, x):
        x = x.reshape(self.array_shape)
        _x, _phi = self.separate_state(x)
        _x = _x @ self.p_k.weights
        return _x.reshape(-1)

    def normalize_state(self, x, overclip=True):
        _x = x.copy()
        _x, _phi = self.separate_state(_x)
        _x = self.__normalize_x(_x, overclip)
        _phi = self.__normalize_phi(_phi, overclip)
        normed_x = self.group_state(_x, _phi)
        return normed_x

    def __normalize_x(self, x, overclip=True):
        x[x <= 0] = EPSILON
        if overclip:
            x[x >= 1] = 1 - EPSILON
        z = x.sum(0)
        return x / z

    def __normalize_phi(self, phi, overclip=True):
        phi[phi <= 0] = EPSILON
        if overclip:
            phi[phi >= 1] = 1 - EPSILON
        z = phi.sum(1)
        for i in range(self.s_dim):
            phi[i] /= z[i]
        return phi

    def separate_state(self, x):
        _x = x[: self.s_dim]
        _phi = x[self.s_dim :].reshape(self.s_dim, self.s_dim, -1)
        return _x, _phi

    def group_state(self, x, phi):
        grouped_x = np.zeros(self.array_shape)
        grouped_x[: self.s_dim] = x
        grouped_x[self.s_dim :] = phi.reshape(self.s_dim * self.s_dim, -1)
        return grouped_x

    def __random_state(self):
        x = np.random.rand(self.s_dim, self.k_dim)
        return self.__normalize_x(x)

    def __random_phi(self):
        phi = np.random.rand(self.s_dim, self.s_dim, self.k_dim)
        return self.__normalize_phi(phi)

    def __marginalize_on_l(self, x):
        _x = x.copy()
        _x.T[self.bad_config.T] = 0
        for i in range(self.s_dim):
            _x = _x.sum(-1)
        return _x

    def get_phi(self, x):
        dist = lambda x, y: np.abs(x - y).mean()
        diff = np.inf
        phi = self.__random_phi()
        while diff > self.tol:
            _phi = self.app_phi(x, phi)
            diff = dist(phi, _phi)
            phi = _phi
        return phi
