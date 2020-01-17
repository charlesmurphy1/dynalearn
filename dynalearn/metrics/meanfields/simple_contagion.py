from .mf import *


class SIS_MF(MF):
    def __init__(self, degree_dist, params, tol=1e-3, verbose=0, dtype="float"):
        self.params = params
        super(SIS_MF, self).__init__(
            2, degree_dist, tol=tol, verbose=verbose, dtype=dtype
        )

    def compute_ltp(self,):
        ltp = np.zeros((self.s_dim, self.s_dim, *self.k_grid.shape))
        ltp[0, 0] = (1 - self.params["infection"]) ** self.l_grid[1]
        ltp[0, 1] = 1 - ltp[0, 0]
        ltp[1, 0] = self.params["recovery"] * np.ones(self.k_grid.shape)
        ltp[1, 1] = 1 - ltp[1, 0]
        return ltp


class SIR_MF(MF):
    def __init__(self, degree_dist, params, tol=1e-3, verbose=0, dtype="float"):
        self.params = params
        super(SIR_MF, self).__init__(
            3, degree_dist, tol=tol, verbose=verbose, dtype=dtype
        )

    def compute_ltp(self,):
        ltp = np.zeros((self.s_dim, self.s_dim, *self.k_grid.shape))
        ltp[0, 0] = (1 - self.params["infection"]) ** self.l_grid[1]
        ltp[0, 1] = 1 - ltp[0, 0]
        ltp[0, 2] = 0
        ltp[1, 0] = 0
        ltp[1, 1] = 1 - self.params["recovery"] * np.ones(self.k_grid.shape)
        ltp[1, 2] = 1 - ltp[1, 1]
        ltp[1, 0] = 0
        ltp[1, 1] = 0
        ltp[1, 2] = 1
        return ltp
