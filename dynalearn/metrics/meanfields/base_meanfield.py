import numpy as np

from scipy.optimize import approx_fprime
from .finder import RecurrenceFPF
from .utilities import power_method, EPSILON
import tqdm


class BaseMeanField:
    def __init__(self, array_shape, tol=1e-5, verbose=0, dtype="float"):
        self.array_shape = array_shape
        self.tol = tol
        self.verbose = verbose
        self.dtype = dtype

        self.ltp = self.compute_ltp()  # m x i x j
        if self.verbose:
            print(f"System size: {np.prod(self.array_shape)}")

    def random_state(self):
        x = np.random.rand(*self.array_shape).astype(self.dtype)
        x = self.normalize_state(x)
        return x.reshape(-1)

    def search_fixed_point(self, x0=None, fp_finder=None):
        if fp_finder is None:
            fp_finder = RecurrenceFPF(
                tol=self.tol, max_iter=10000, verbose=self.verbose
            )
        if x0 is None:
            _x0 = self.random_state()
        else:
            _x0 = self.normalize_state(x0.reshape(self.array_shape)).reshape(-1)
        sol = fp_finder(self.application, _x0)
        if not sol.success:
            print("Converge warning: Still returning final result.")
        return sol.x

    def stability(self, fp, epsilon=1e-6):
        jac = self.approx_jacobian(fp, epsilon=epsilon)
        w = np.linalg.eigvals(jac)
        return np.max(np.abs(w))

    def isclose(self, x, y):
        dist = np.sum(np.abs(x - y))
        return dist < 1e-2

    def clip(self, x):
        x[x <= 0] = EPSILON
        x[x >= 1] = 1 - EPSILON
        return x

    def approx_jacobian(self, x, epsilon=1e-6):
        jac = np.zeros((x.shape[0], x.shape[0])).astype(self.dtype)
        if self.verbose != 0:
            print("Computing Jacobian matrix")
        if self.verbose == 1:
            pb = tqdm.tqdm(range(x.shape[0]))
        for i in range(x.shape[0]):
            f = lambda xx: self.application(xx)[i]
            jac[i] = approx_fprime(x, f, epsilon)
            if self.verbose == 1:
                pb.update()
        if self.verbose == 1:
            pb.close()
        return jac

    def compute_thresholds(self):
        raise NotImplementedError()

    def compute_ltp(self):
        raise NotImplementedError()

    def application(self, x):
        raise NotImplementedError()

    def to_compartment(self, graph, state):
        raise NotImplementedError()

    def to_avg(self, x):
        raise NotImplementedError()

    def normalize_state(self, x, clip=True):
        raise NotImplementedError()
