import networkx as nx
import numpy as np
import tqdm

from abc import ABC, abstractmethod
from dynalearn.utilities import (
    all_combinations,
    numba_all_combinations,
    numba_logfactorial,
    numba_multinomial,
    to_nary,
)
from itertools import product
from numba import jit
from scipy.special import gammaln
from scipy.stats import multinomial


def _marginal_ltp(x, y, k, phi, ltp):
    mltp = 0
    phi /= phi.sum()
    if k == 0:
        return ltp[0, x, y]
    dist = multinomial(k, phi)
    for i, ll in enumerate(all_combinations(k, len(phi))):
        mltp += ltp[i, x, y] * dist.pmf(ll)
    return mltp


@jit(nopython=True)
def _numba_marginal_ltp(x, y, k, phi, ltp):
    mltp = 0
    phi /= phi.sum()
    if k == 0:
        return ltp[0, x, y]
    for i, ll in enumerate(numba_all_combinations(k, len(phi))):
        mltp += ltp[i, x, y] * numba_multinomial(k, ll, phi)
    return mltp


class Meanfield(ABC):
    def __init__(self, p_k, num_states, with_numba=False):
        self.num_states = num_states
        self.ltp = None
        self.with_numba = with_numba
        self.p_k = p_k

    def compute_ltp(self):
        return

    def marginal_ltp(self, x, y, k, phi):
        if self.with_numba:
            return _numba_marginal_ltp(x, y, k, phi, self.ltp[k])
        else:
            return _marginal_ltp(x, y, k, phi, self.ltp[k])

    def avg(self, x):
        avg_x = np.zeros(self.num_states)
        for i, k in enumerate(self.k):
            for j in range(self.num_states):
                avg_x[j] += x[k][j] * self.p_k.weights[i]
        return avg_x

    def flatten(self, x):
        return np.concatenate([x[k] for k in self.k])

    def unflatten(self, flat_x):
        return {
            k: flat_x[i * self.num_states : (i + 1) * self.num_states]
            for i, k in enumerate(self.k)
        }

    def normalize_state(self, x):
        y = x.copy()
        for k in self.k:
            y[k] /= y[k].sum()
        return y

    def phi(self, x):
        avg_k = 0
        avg_xk = np.zeros(self.num_states)
        for (i, k), j in product(enumerate(self.k), range(self.num_states)):
            avg_xk[j] += x[k][j] * k * self._p_k.weights[i]
        return avg_xk / self.avg_k

    def random_state(self):
        x = {k: np.random.rand(self.num_states) for k in self.k}
        x = self.normalize_state(x)
        return x

    def update(self, x):
        y = {k: np.zeros(self.num_states) for k in self.k}
        phi = self.phi(x)

        for k, i, j in product(self.k, range(self.num_states), range(self.num_states)):
            y[k][i] += self.marginal_ltp(j, i, k, phi) * x[k][j]
        y = self.normalize_state(y)
        return y

    @property
    def p_k(self):
        if self._p_k is None:
            raise NotImplementedError()
        return self._p_k

    @p_k.setter
    def p_k(self, p_k):
        self._p_k = p_k
        self.k = p_k.values
        self.k_min = self.p_k.values.min()
        self.k_max = self.p_k.values.max()
        self.k_dim = int(self.k_max - self.k_min + 1)
        self.avg_k = 0
        for (i, k) in enumerate(self.k):
            self.avg_k += k * self._p_k.weights[i]
        self.shape = (self.k_dim, self.num_states)
        self.compute_ltp()


class GenericMeanfield(Meanfield):
    def __init__(self, p_k, model, with_numba=True):
        self.model = model
        if "window_size" in self.model.__dict__:
            self.window_size = self.model.window_size
        else:
            self.window_size = 1
        if self.model.__class__.__name__ == "TrainableEpidemics":
            num_states = self.model.num_states ** self.model.window_size
        else:
            num_states = model.num_states
        Meanfield.__init__(self, p_k, num_states, with_numba=with_numba)

    def update(self, x):
        y = {k: np.zeros(self.num_states) for k in self.k}
        phi = self.phi(x)

        for k, i, j in product(self.k, range(self.num_states), range(self.num_states)):
            b_i = to_nary(i, base=self.model.num_states, dim=self.window_size).squeeze()
            b_j = to_nary(j, base=self.model.num_states, dim=self.window_size).squeeze()
            if len(b_j.shape) > 0 and len(b_i.shape) > 0:
                if np.all(b_j[1:] == b_i[:-1]):
                    i = int(b_i[-1])
                    y[k][i] += self.marginal_ltp(j, i, k, phi) * x[k][j]
            else:
                y[k][i] += self.marginal_ltp(j, i, k, phi) * x[k][j]

        y = self.normalize_state(y)
        return y

    def compute_ltp(self):
        ltp = {}
        i = 0
        for k in self.p_k.values:
            neighbor_states = np.array(all_combinations(k, self.num_states))
            _ltp = np.zeros(
                (neighbor_states.shape[0], self.num_states, self.model.num_states)
            )
            g = nx.empty_graph(self.p_k.values.max() + 1)
            g.add_edges_from(nx.star_graph(k + 1).edges())
            self.model.network = g
            for i, ns in enumerate(neighbor_states):
                x = np.zeros(g.number_of_nodes())
                y = np.concatenate([ss * np.ones(ll) for ss, ll in enumerate(ns)])
                x[1 : y.shape[0] + 1] = y * 1
                for s in range(self.num_states):
                    x[0] = s
                    bin_x = to_nary(x, base=self.num_states, dim=self.window_size)
                    _ltp[i, s] = self.model.predict(bin_x)[0]
            ltp[k] = _ltp
        self.ltp = ltp
