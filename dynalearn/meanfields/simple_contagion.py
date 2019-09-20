from .ame import *
from .pa import *
from .mf import *


class SIS_AME(AME):
    def __init__(self, p_k, infection_prob, recovery_prob, tol=1e-3, verbose=1):
        self.infection_prob = infection_prob
        self.recovery_prob = recovery_prob
        super(SIS_AME, self).__init__(2, p_k, tol=tol, verbose=verbose)

    def compute_ltp(self,):
        ltp = np.zeros((self.s_dim, self.s_dim, *self.k_grid.shape))
        ltp[0, 0] = (1 - self.infection_prob) ** self.l_grid[1]
        ltp[0, 1] = 1 - ltp[0, 0]
        ltp[1, 0] = self.recovery_prob * np.ones(self.k_grid.shape)
        ltp[1, 1] = 1 - ltp[1, 0]
        return ltp


class SIS_PA(PA):
    def __init__(self, p_k, infection_prob, recovery_prob, tol=1e-3, verbose=1):
        self.infection_prob = infection_prob
        self.recovery_prob = recovery_prob
        super(SIS_PA, self).__init__(2, p_k, tol=tol, verbose=verbose)

    def compute_ltp(self,):
        ltp = np.zeros((self.s_dim, self.s_dim, *self.k_grid.shape))
        ltp[0, 0] = (1 - self.infection_prob) ** self.l_grid[1]
        ltp[0, 1] = 1 - ltp[0, 0]
        ltp[1, 0] = self.recovery_prob * np.ones(self.k_grid.shape)
        ltp[1, 1] = 1 - ltp[1, 0]
        return ltp


class SIS_MF(MF):
    def __init__(self, p_k, infection_prob, recovery_prob, tol=1e-3, verbose=1):
        self.infection_prob = infection_prob
        self.recovery_prob = recovery_prob
        super(SIS_MF, self).__init__(2, p_k, tol=tol, verbose=verbose)

    def compute_ltp(self,):
        ltp = np.zeros((self.s_dim, self.s_dim, *self.k_grid.shape))
        ltp[0, 0] = (1 - self.infection_prob) ** self.l_grid[1]
        ltp[0, 1] = 1 - ltp[0, 0]
        ltp[1, 0] = self.recovery_prob * np.ones(self.k_grid.shape)
        ltp[1, 1] = 1 - ltp[1, 0]
        return ltp


class SIR_AME(AME):
    def __init__(self, p_k, infection_prob, recovery_prob, tol=1e-3, verbose=1):
        self.infection_prob = infection_prob
        self.recovery_prob = recovery_prob
        super(SIR_AME, self).__init__(3, p_k, tol=tol, verbose=verbose)

    def compute_ltp(self,):
        ltp = np.zeros((self.s_dim, self.s_dim, *self.k_grid.shape))
        ltp[0, 0] = (1 - self.infection_prob) ** self.l_grid[1]
        ltp[0, 1] = 1 - ltp[0, 0]
        ltp[0, 2] = 0
        ltp[1, 0] = 0
        ltp[1, 1] = 1 - self.recovery_prob * np.ones(self.k_grid.shape)
        ltp[1, 2] = 1 - ltp[1, 1]
        ltp[1, 0] = 0
        ltp[1, 1] = 0
        ltp[1, 2] = 1
        return ltp


class SIR_PA(PA):
    def __init__(self, p_k, infection_prob, recovery_prob, tol=1e-3, verbose=1):
        self.infection_prob = infection_prob
        self.recovery_prob = recovery_prob
        super(SIR_PA, self).__init__(3, p_k, tol=tol, verbose=verbose)

    def compute_ltp(self,):
        ltp = np.zeros((self.s_dim, self.s_dim, *self.k_grid.shape))
        ltp[0, 0] = (1 - self.infection_prob) ** self.l_grid[1]
        ltp[0, 1] = 1 - ltp[0, 0]
        ltp[0, 2] = 0
        ltp[1, 0] = 0
        ltp[1, 1] = 1 - self.recovery_prob * np.ones(self.k_grid.shape)
        ltp[1, 2] = 1 - ltp[1, 1]
        ltp[1, 0] = 0
        ltp[1, 1] = 0
        ltp[1, 2] = 1
        return ltp


class SIR_MF(MF):
    def __init__(self, p_k, infection_prob, recovery_prob, tol=1e-3, verbose=1):
        self.infection_prob = infection_prob
        self.recovery_prob = recovery_prob
        super(SIR_MF, self).__init__(3, p_k, tol=tol, verbose=verbose)

    def compute_ltp(self,):
        tp = np.zeros((self.s_dim, self.s_dim, *self.k_grid.shape))
        ltp[0, 0] = (1 - self.infection_prob) ** self.l_grid[1]
        ltp[0, 1] = 1 - ltp[0, 0]
        ltp[0, 2] = 0
        ltp[1, 0] = 0
        ltp[1, 1] = 1 - self.recovery_prob * np.ones(self.k_grid.shape)
        ltp[1, 2] = 1 - ltp[1, 1]
        ltp[1, 0] = 0
        ltp[1, 1] = 0
        ltp[1, 2] = 1
        return ltp
