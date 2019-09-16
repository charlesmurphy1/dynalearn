import numpy as np
from scipy.optimize import root, approx_fprime
import tqdm


class BaseMeanField:
    def __init__(self, array_shape, max_iter=100, tol=1e-3, verbose=1):
        self.array_shape = array_shape
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        self.fixed_points = []
        self.ltp = self.compute_ltp()  # m x i x j

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
        if len(self.fixed_points) == 0:
            self.fixed_points.append(x)
        elif (
            np.sum(
                np.prod(np.isclose(np.array(self.fixed_points), x, atol=self.tol), -1)
            )
            == 0
        ):
            self.fixed_points.append(x)

    def compute_fixed_points(self, num_seeds=1, check_abs_states=True):
        def f_to_solve(x):
            y = self.application(x) - x
            return y

        if self.verbose:
            pb = tqdm.tqdm(range(num_seeds), "Finding fixed points")
        for i in range(num_seeds):
            x0 = self.random_state()
            sol = root(f_to_solve, x0)
            if sol.success:
                self.add_fixed_points(sol.x)
            if self.verbose:
                pb.update()
        if self.verbose:
            pb.close()

        if check_abs_states:
            for i in range(self.s_dim):
                abs_state = self.abs_state(i)
                sol = root(f_to_solve, abs_state)
                if sol.success:
                    self.add_fixed_points(sol.x)

    def approx_jaccobian(self, x, epsilon=1e-6):
        jac = np.zeros((x.shape[0], x.shape[0]))
        for i in range(x.shape[0]):
            f = lambda xx: self.application(xx)[i]
            jac[i] = approx_fprime(x, f, epsilon)
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
