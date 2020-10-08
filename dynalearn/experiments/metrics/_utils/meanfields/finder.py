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
    def __init__(self, tol, verbose):
        self.tol = tol
        self.verbose = verbose

    def has_converged(self, x0, x1):
        diff = np.abs(x0 - x1)
        if np.any(diff > self.tol):
            return False
        else:
            return True

    def __call__(self, f_to_solve, **kwargs):
        raise NotImplementedError()


class RecurrenceFPF(FixedPointFinder):
    def __init__(self, tol=1e-6, initial_iter=100, max_iter=1000, verbose=0):
        self.initial_iter = initial_iter
        self.max_iter = max_iter
        FixedPointFinder.__init__(self, tol, verbose)

    def __call__(self, f, x0):
        for i in range(self.max_iter):
            x1 = f(x0)
            if self.has_converged(x0, x1) and i > self.initial_iter:
                return FinderResult(x1, np.abs(x0 - x1), True, i)
            x0 = x1 * 1

        if self.verbose != 0:
            print(f"No further progress after {i} evaluations.")
        return FinderResult(x1, np.abs(x0 - x1), False, i)


class NewtonFPF(FixedPointFinder):
    def __init__(self, tol=1e-6, max_iter=1000, verbose=0):
        self.max_iter = max_iter
        FixedPointFinder.__init__(self, tol, verbose)

    def __call__(self, f, x0):
        f_to_solve = lambda x: f(x) - x
        sol = root(f_to_solve, x0, tol=self.tol)
        return FinderResult(sol.x, f_to_solve(sol.x), sol.success, sol.nfev)


class HybridRecurrenceNewtonFPF(FixedPointFinder):
    def __init__(self, tol=1e-6, rec_iter=100, max_iter=1000, verbose=0):
        self.rec_iter = rec_iter
        self.max_iter = max_iter
        FixedPointFinder.__init__(self, tol, verbose)

    def __call__(self, f, x0):
        f_to_solve = lambda x: f(x) - x
        for i in range(self.max_iter):
            if i % self.rec_iter == 0:
                sol = root(f_to_solve, x0, tol=self.tol)
                if self.has_converged(sol.x, x0):
                    return FinderResult(sol.x, f_to_solve(sol.x), True, i)
            x1 = f(x0)
            if self.has_converged(x0, x1):
                return FinderResult(x1, f_to_solve(x1), True, i)
            x0 = x1 * 1
        if self.verbose != 0:
            print(f"No further progress after {i} evaluations.")
        return FinderResult(x0, f_to_solve(x0), False, i)


__finders__ = {
    "RecurrenceFPF": lambda config: RecurrenceFPF(
        tol=config.tol, initial_iter=config.initial_iter, max_iter=config.max_iter
    ),
    "NewtonFPF": lambda config: NewtonFPF(tol=config.tol, max_iter=config.max_iter),
    "HybridRecurrenceNewtonFPF": lambda config: HybridRecurrenceNewtonFPF(
        tol=config.tol, rec_iter=config.rec_iter, max_iter=config.max_iter
    ),
}


def get(config):
    name = config.finder
    if name in __finders__:
        return __finders__[name](config)
    else:
        raise ValueError(
            f"{name} is invalid, valid entries are {list(__finders__.keys())}."
        )
