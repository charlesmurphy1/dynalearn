import numpy as np
from scipy.optimize import root, newton


class FinderResult(object):
    def __init__(self, x, func, success, nfev):
        super(FinderResult, self).__init__()
        self.x = x
        self.func = func
        self.success = success
        self.nfev = nfev

    def __repr__(self,):
        s = f"      x: {self.x}\n"
        s += f"   func: {self.func}\n"
        s += f"success: {self.success}\n"
        s += f"   nfev: {self.nfev}\n"
        return s


class FixedPointFinder(object):
    def __init__(self, verbose):
        self.verbose = verbose

    def dist(self, x, y):
        return np.sqrt(((x - y) ** 2).sum())

    def __call__(self, f_to_solve, **kwargs):
        raise NotImplementedError()


class RecurrenceFPF(FixedPointFinder):
    def __init__(self, tol=1e-6, max_iter=1000, verbose=1):
        self.tol = tol
        self.max_iter = max_iter
        super(RecurrenceFPF, self).__init__(verbose)

    def __call__(self, f, x0):
        nfev = 0
        diff = np.inf
        success = True
        while diff > self.tol:
            _x = f(x0)
            diff = self.dist(x0, _x)
            x0 = _x * 1
            nfev += 1
            if nfev > self.max_iter:
                success = False
                if self.verbose:
                    print(f"No further progress after {nfev} evaluations.")
                break
        return FinderResult(x0, x0 - f(x0), success, nfev)


class ApproxNewtonFPF(FixedPointFinder):
    def __init__(self, tol=1e-6, max_iter=1000, verbose=1):
        self.tol = tol
        self.max_iter = max_iter
        super(ApproxNewtonFPF, self).__init__(verbose)

    def __call__(self, f, x0):
        f_to_solve = lambda x: f(x) - x
        sol = root(f, x0)
        return FinderResult(sol.x, f_to_solve(sol.x), sol.success, sol.nfev)
