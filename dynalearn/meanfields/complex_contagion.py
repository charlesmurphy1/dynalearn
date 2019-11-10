from .ame import *
from .pa import *
from .mf import *
import numpy as np


def sigmoid(x):
    return 1.0 / (np.exp(-x) + 1)


def st_infprob(mu, beta, l_grid, k_grid):
    x = l_grid[1] / k_grid
    return sigmoid(beta * (x - mu))


class SoftThresholdSIS_MF(MF):
    def __init__(
        self, p_k, mu, beta, recovery_prob, tol=1e-3, verbose=1, dtype="float"
    ):
        self.mu = mu
        self.beta = beta
        self.recovery_prob = recovery_prob
        super(SoftThresholdSIS_MF, self).__init__(
            2, p_k, tol=tol, verbose=verbose, dtype=dtype
        )

    def compute_ltp(self,):
        ltp = np.zeros((self.s_dim, self.s_dim, *self.k_grid.shape))
        inf_prob = st_infprob(self.mu, self.beta, self.l_grid, self.k_grid)
        ltp[0, 0] = 1 - inf_prob
        ltp[0, 1] = inf_prob
        ltp[1, 0] = self.recovery_prob * np.ones(self.k_grid.shape)
        ltp[1, 1] = 1 - ltp[1, 0]
        return ltp


class SoftThresholdSIR_MF(MF):
    def __init__(
        self, p_k, mu, beta, recovery_prob, tol=1e-3, verbose=1, dtype="float"
    ):
        self.mu = mu
        self.beta = beta
        self.recovery_prob = recovery_prob
        super(SoftThresholdSIR_MF, self).__init__(
            3, p_k, tol=tol, verbose=verbose, dtype=dtype
        )

    def compute_ltp(self,):
        ltp = np.zeros((self.s_dim, self.s_dim, *self.k_grid.shape))
        inf_prob = st_infprob(self.mu, self.beta, self.l_grid, self.k_grid)
        ltp[0, 0] = 1 - inf_prob
        ltp[0, 1] = inf_prob
        ltp[0, 2] = 0
        ltp[1, 0] = 0
        ltp[1, 1] = 1 - self.recovery_prob * np.ones(self.k_grid.shape)
        ltp[1, 2] = 1 - ltp[1, 1]
        ltp[2, 0] = 0
        ltp[2, 1] = 0
        ltp[2, 2] = 1
        return ltp


def nl_infprob(trans_prob, alpha, l_grid):
    return (1 - (1 - trans_prob)**l_grid)**alpha


class NonLinearSIS_MF(MF):
    def __init__(
        self, p_k, trans_prob, recovery_prob, alpha, tol=1e-3, verbose=1, dtype="float"
    ):
        self.trans_prob = trans_prob
        self.recovery_prob = recovery_prob
        self.alpha = alpha
        super(NonLinearSIS_MF, self).__init__(
            2, p_k, tol=tol, verbose=verbose, dtype=dtype
        )

    def compute_ltp(self,):
        ltp = np.zeros((self.s_dim, self.s_dim, *self.k_grid.shape))
        inf_prob = nl_infprob(self.trans_prob, self.beta, self.l_grid, self.k_grid)
        ltp[0, 0] = 1 - inf_prob
        ltp[0, 1] = inf_prob
        ltp[1, 0] = self.recovery_prob * np.ones(self.k_grid.shape)
        ltp[1, 1] = 1 - ltp[1, 0]
        return ltp


class NonLinearSIR_MF(MF):
    def __init__(
        self, p_k, trans_prob, recovery_prob, alpha, tol=1e-3, verbose=1, dtype="float"
    ):
        self.trans_prob = trans_prob
        self.recovery_prob = recovery_prob
        self.alpha = alpha
        super(NonLinearSIR_MF, self).__init__(
            3, p_k, tol=tol, verbose=verbose, dtype=dtype
        )

    def compute_ltp(self,):
        ltp = np.zeros((self.s_dim, self.s_dim, *self.k_grid.shape))
        inf_prob = nl_infprob(self.trans_prob, self.alpha, self.l_grid, self.k_grid)
        ltp[0, 0] = 1 - inf_prob
        ltp[0, 1] = inf_prob
        ltp[0, 2] = 0
        ltp[1, 0] = 0
        ltp[1, 1] = 1 - self.recovery_prob * np.ones(self.k_grid.shape)
        ltp[1, 2] = 1 - ltp[1, 1]
        ltp[2, 0] = 0
        ltp[2, 1] = 0
        ltp[2, 2] = 1
        return ltp
