from .ame import *
from .pa import *
from .mf import *


class SISSIS_MF(MF):
    def __init__(self, degree_dist, params, tol=1e-3, verbose=0):
        self.params = params
        super(SISSIS_MF, self).__init__(4, degree_dist, tol=tol, verbose=verbose)

    def compute_ltp(self,):
        ltp = np.zeros((self.s_dim, self.s_dim, *self.k_grid.shape))
        tau1, tau2 = self.params["infection1"], self.params["infection2"]
        gamma1, gamma2 = self.params["recovery1"], self.params["recovery2"]
        zeta = self.params["coupling"]
        p0, p1 = (1 - tau1) ** self.l_grid[1], (1 - tau2) ** self.l_grid[2]
        r0, r1 = (
            (1 - zeta * tau1) ** self.l_grid[1],
            (1 - zeta * tau2) ** self.l_grid[2],
        )
        q0, q1 = (
            (1 - zeta * tau1) ** self.l_grid[3],
            (1 - zeta * tau2) ** self.l_grid[3],
        )
        ltp[0, 0] = p0 * q0 * p1 * q1
        ltp[0, 1] = (1 - p0 * q0) * p1 * q1
        ltp[0, 2] = p0 * q0 * (1 - p1 * q1)
        ltp[0, 3] = (1 - p0 * q0) * (1 - p1 * q1)

        ltp[1, 0] = g0 * p1 * q1
        ltp[1, 1] = (1 - g0) * p1 * q1
        ltp[1, 2] = g0 * (1 - p1 * q1)
        ltp[1, 3] = (1 - g0) * (1 - p1 * q1)

        ltp[2, 0] = p0 * q0 * g1
        ltp[2, 1] = (1 - p0 * q0) * g1
        ltp[2, 2] = p0 * q0 * (1 - g1)
        ltp[2, 3] = (1 - p0 * q0) * (1 - g1)

        ltp[3, 0] = g0 * g1
        ltp[3, 1] = (1 - g0) * g1
        ltp[3, 2] = g0 * (1 - g1)
        ltp[3, 3] = (1 - g0) * (1 - g1)

        return ltp
