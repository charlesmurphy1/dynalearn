from .ame import *
from .pa import *
from .mf import *


class SISSIS_AME(AME):
    def __init__(self, p_k, infection_prob, recovery_prob, coupling, tol=1e-3, verbose=1):
        self.infection_prob = infection_prob
        self.recovery_prob = recovery_prob
        self.coupling = coupling
        super(SISSIS_AME, self).__init__(4, p_k, tol=tol, verbose=verbose)

    def compute_ltp(self,):
        ltp = np.zeros((self.s_dim, self.s_dim, *self.k_grid.shape))
        t0, t1 = self.infection_prob
        g0, g1 = self.recovery_prob
        c = self.coupling
        p0, p1 = (1 - t0)**self.l_grid[1], (1 - t1)**self.l_grid[2]
        r0, r1 = (1 - c*t0)**self.l_grid[1], (1 - c*t1)**self.l_grid[2]
        q0, q1 = (1 - c*t0)**self.l_grid[3], (1 - c*t1)**self.l_grid[3]
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



class SISSIS_MF(MF):
    def __init__(self, p_k, infection_prob, recovery_prob, coupling, tol=1e-3, verbose=1):
        self.infection_prob = infection_prob
        self.recovery_prob = recovery_prob
        self.coupling = coupling
        super(SISSIS_MF, self).__init__(4, p_k, tol=tol, verbose=verbose)

    def compute_ltp(self,):
        ltp = np.zeros((self.s_dim, self.s_dim, *self.k_grid.shape))
        t0, t1 = self.infection_prob
        g0, g1 = self.recovery_prob
        c = self.coupling
        p0, p1 = (1 - t0)**self.l_grid[1], (1 - t1)**self.l_grid[2]
        r0, r1 = (1 - c*t0)**self.l_grid[1], (1 - c*t1)**self.l_grid[2]
        q0, q1 = (1 - c*t0)**self.l_grid[3], (1 - c*t1)**self.l_grid[3]
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
