from .mf import *


class SIS_MF(MF):
    def __init__(self, p_k, params, tol=1e-3, verbose=1, dtype="float"):
        super(SIS_MF, self).__init__(
            2, p_k, params, tol=tol, verbose=verbose, dtype=dtype
        )

    def compute_ltp(self,):
        ltp = np.zeros((self.s_dim, self.s_dim, *self.k_grid.shape))
        ltp[0, 0] = (1 - self.params["infection"]) ** self.l_grid[1]
        ltp[0, 1] = 1 - ltp[0, 0]
        ltp[1, 0] = self.self.params["recovery"] * np.ones(self.k_grid.shape)
        ltp[1, 1] = 1 - ltp[1, 0]
        return ltp


class SIR_MF(MF):
    def __init__(self, p_k, params, tol=1e-3, verbose=1, dtype="float"):
        super(SIR_MF, self).__init__(3, p_k, tol=tol, verbose=verbose, dtype=dtype)

    def compute_ltp(self,):
        tp = np.zeros((self.s_dim, self.s_dim, *self.k_grid.shape))
        ltp[0, 0] = (1 - self.params["infection"]) ** self.l_grid[1]
        ltp[0, 1] = 1 - ltp[0, 0]
        ltp[0, 2] = 0
        ltp[1, 0] = 0
        ltp[1, 1] = 1 - self.self.params["recovery"] * np.ones(self.k_grid.shape)
        ltp[1, 2] = 1 - ltp[1, 1]
        ltp[1, 0] = 0
        ltp[1, 1] = 0
        ltp[1, 2] = 1
        return ltp
