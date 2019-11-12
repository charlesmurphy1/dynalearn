import numpy as np
from scipy.special import gammaln


class DiscreteDistribution(object):
    def __init__(self, values):
        super(DiscreteDistribution, self).__init__()
        self.val_dict = {k: v for k, v in zip(*values)}
        self.values = values[0]  # Size K
        self.weights = values[1]  # Size K

    def expect(self, func):
        x = func(self.k)  # Size K x D
        return self.p_k @ x

    def mean(self):
        f = lambda k: k
        return self.expect(f)

    def var(self):
        f = lambda k: (k - self.mean()) ^ 2
        return self.expect(f)

    def std(self):
        return np.sqrt(self.var())


def kronecker_distribution(k):
    k = np.array([k])
    p_k = np.array([1])
    return DiscreteDistribution((k, p_k))


def poisson_distribution(avgk, k=None, num_k=3, tol=None):
    f = lambda k: np.exp(-avgk + k * np.log(avgk) - gammaln(k + 1))
    if k is not None:
        p_k = f(k)
        return DiscreteDistribution((k, p_k))
    if tol is None:
        mid_k = np.round(avgk)
        if mid_k < num_k:
            down = 1
            up = 2 * num_k + 2
        else:
            down = mid_k - num_k + 1
            up = mid_k + num_k + 2
        k = np.arange(down, up).astype("int")
        p_k = f(k)
        p_k /= np.sum(p_k)
        return DiscreteDistribution((k, p_k))
    else:
        k = []
        p_k = []
        k0 = 1
        while len(k) < 1 or dist > tol:
            dist = f(k0)
            if dist > tol:
                k.append(k0)
                p_k.append(dist)
            k0 += 1
        k = np.array(k)
        p_k = np.array(p_k)
        p_k /= np.sum(p_k)
        return DiscreteDistribution((k, p_k))