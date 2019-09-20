import numpy as np

from scipy.optimize import approx_fprime
from .fixed_point import RecurrenceFPF
from .utilities import power_method
import tqdm


class BaseMeanField:
    def __init__(self, array_shape, tol=1e-3, verbose=1):
        self.array_shape = array_shape
        self.tol = tol
        self.verbose = verbose

        self.fixed_points = []
        self.stability = []
        self.ltp = self.compute_ltp()  # m x i x j
        if self.verbose:
            print(f"System size: {np.prod(self.array_shape)}")

    def random_state(self):
        x = np.random.rand(*self.array_shape)
        x = self.normalize_state(x)
        return x.reshape(-1)

    def abs_state(self, s):
        x = np.zeros(self.array_shape)
        x[s] = 1
        x = self.normalize_state(x)
        return x.reshape(-1)

    def add_fixed_points(self, x):
        isclose = False
        if len(self.fixed_points) == 0:
            self.fixed_points.append(x)
            return

        for fp in self.fixed_points:
            if self.isclose(fp, x):
                return
        self.fixed_points.append(x)

    def compute_fixed_points(self, fp_finder=None, num_seeds=1, x0=None, epsilon=1e-6):
        if fp_finder is None:
            fp_finder = RecurrenceFPF(
                tol=self.tol, max_iter=10000, verbose=self.verbose
            )
        if self.verbose:
            pb = tqdm.tqdm(range(num_seeds), "Finding fixed points")
        for i in range(num_seeds):
            if x0 is None:
                _x0 = self.random_state()
            else:
                _x0 = x0 + epsilon * np.random.randn(*x0.shape)
                _x0 = self.normalize_state(_x0.reshape(self.array_shape)).reshape(-1)
            sol = fp_finder(self.application, _x0)
            if sol.success:
                self.add_fixed_points(sol.x)
            if self.verbose:
                pb.update()
        if self.verbose:
            pb.close()

    def compute_stability(self, epsilon=1e-6):
        for fp in self.fixed_points:
            jac = self.approx_jacobian(x, epsilon=epsilon)
            w, v = power_method(jac, self.tol, max_iter=1000)
            self.stability.append(np.abs(w))

    def isclose(self, x, y):
        dist = np.sum(np.abs(x - y))
        return dist < 1e-2

    def approx_jacobian(self, x, epsilon=1e-6):
        jac = np.zeros((x.shape[0], x.shape[0]))
        if self.verbose:
            pb = tqdm.tqdm(range(x.shape[0]), "Computing Jacobian matrix")
        for i in range(x.shape[0]):
            f = lambda xx: self.application(xx)[i]
            jac[i] = approx_fprime(x, f, epsilon)
            if self.verbose:
                pb.update()
        if self.verbose:
            pb.close()
        return jac

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
