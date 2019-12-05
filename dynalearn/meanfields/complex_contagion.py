from .ame import *
from .pa import *
from .mf import *
import numpy as np
from scipy.special import lambertw


def sigmoid(x):
    return 1.0 / (np.exp(-x) + 1)


def constant_deactivation(l_grid, k_grid, params):
    return params["recovery"] * np.ones(self.k_grid.shape)


def soft_threshold_activation(l_grid, k_grid, params):
    x = l_grid[1]
    return sigmoid(params["slope"] * (x - params["threshold"]))


def nonlinear_activation(l_grid, k_grid, params):
    l = l_grid[1]
    return (1 - (1 - params["infection"]) ** l) ** params["exponent"]


def sine_activation(l_grid, k_grid, params):
    l = l_grid[1]
    return (1 - (1 - params["infection"]) ** l) * (
        1 - params["amplitude"] * (np.sin(np.pi * l / params["period"])) ** 2
    )


def planck_activation(l_grid, k_grid, params):
    l = l_grid[1]
    gamma = (lambertw(-3 * np.exp(-3)) + 3).real
    Z = gamma ** 3 * params["temperature"] ** 3 / (np.exp(gamma) - 1)
    return l ** 3 / (np.exp(l / params["temperature"]) - 1) / Z


class ComplexSISMF(MF):
    def __init__(
        self, activation, deactivation, degree_dist, tol=1e-3, verbose=1, dtype="float"
    ):
        self.activation = activation
        self.deactivation = deactivation
        super(ComplexSISMF, self).__init__(
            2, degree_dist, tol=tol, verbose=verbose, dtype=dtype
        )

    def compute_ltp(self,):
        ltp = np.zeros((self.s_dim, self.s_dim, *self.k_grid.shape))
        activation_prob = self.activation(self.l_grid, self.k_grid)
        deactivation_prob = self.deactivation(self.l_grid, self.k_grid)
        ltp[0, 0] = 1 - activation
        ltp[0, 1] = activation
        ltp[1, 0] = deactivation
        ltp[1, 1] = 1 - deactivation
        return ltp


class ComplexSIRMF(MF):
    def __init__(
        self, activation, deactivation, degree_dist, tol=1e-3, verbose=1, dtype="float"
    ):
        self.activation = activation
        self.deactivation = deactivation
        super(ComplexSIRMF, self).__init__(
            3, degree_dist, tol=tol, verbose=verbose, dtype=dtype
        )

    def compute_ltp(self,):
        ltp = np.zeros((self.s_dim, self.s_dim, *self.k_grid.shape))
        activation_prob = self.activation(self.l_grid, self.k_grid)
        deactivation_prob = self.deactivation(self.l_grid, self.k_grid)
        ltp[0, 0] = 1 - activation_prob
        ltp[0, 1] = activation_prob
        ltp[0, 2] = 0
        ltp[1, 0] = 0
        ltp[1, 1] = 1 - deactivation_prob
        ltp[1, 2] = deactivation_prob
        ltp[2, 0] = 0
        ltp[2, 1] = 0
        ltp[2, 2] = 1
        return ltp


class SoftThresholdSIS_MF(ComplexSISMF):
    def __init__(self, degree_dist, params, tol=1e-3, verbose=1, dtype="float"):
        self.params = params
        activation = lambda l, k: soft_threshold_activation(l, k, params)
        deactivation = lambda l, k: constant_deactivation(l, k, params)

        super(SoftThresholdSIS_MF, self).__init__(
            activation, deactivation, degree_dist, tol=tol, verbose=verbose, dtype=dtype
        )


class SoftThresholdSIR_MF(ComplexSIRMF):
    def __init__(self, degree_dist, params, tol=1e-3, verbose=1, dtype="float"):
        self.params = params
        activation = lambda l, k: soft_threshold_activation(l, k, params)
        deactivation = lambda l, k: constant_deactivation(l, k, params)

        super(SoftThresholdSIR_MF, self).__init__(
            activation, deactivation, degree_dist, tol=tol, verbose=verbose, dtype=dtype
        )


class NonLinearSIS_MF(ComplexSISMF):
    def __init__(self, degree_dist, params, tol=1e-3, verbose=1, dtype="float"):
        self.params = params
        activation = lambda l, k: nonlinear_activation(l, k, params)
        deactivation = lambda l, k: constant_deactivation(l, k, params)

        super(NonLinearSIS_MF, self).__init__(
            activation, deactivation, degree_dist, tol=tol, verbose=verbose, dtype=dtype
        )


class NonLinearSIR_MF(ComplexSIRMF):
    def __init__(self, degree_dist, params, tol=1e-3, verbose=1, dtype="float"):
        self.params = params
        activation = lambda l, k: nonlinear_activation(l, k, params)
        deactivation = lambda l, k: constant_deactivation(l, k, params)

        super(NonLinearSIR_MF, self).__init__(
            activation, deactivation, degree_dist, tol=tol, verbose=verbose, dtype=dtype
        )


class SineSIS_MF(ComplexSISMF):
    def __init__(self, degree_dist, params, tol=1e-3, verbose=1, dtype="float"):
        self.params = params
        activation = lambda l, k: sine_activation(l, k, params)
        deactivation = lambda l, k: constant_deactivation(l, k, params)

        super(SineSIS_MF, self).__init__(
            activation, deactivation, degree_dist, tol=tol, verbose=verbose, dtype=dtype
        )


class SineSIR_MF(ComplexSIRMF):
    def __init__(self, degree_dist, params, tol=1e-3, verbose=1, dtype="float"):
        self.params = params
        activation = lambda l, k: sine_activation(l, k, params)
        deactivation = lambda l, k: constant_deactivation(l, k, params)

        super(SineSIR_MF, self).__init__(
            activation, deactivation, degree_dist, tol=tol, verbose=verbose, dtype=dtype
        )


class PlanckSIS_MF(ComplexSISMF):
    def __init__(self, degree_dist, params, tol=1e-3, verbose=1, dtype="float"):
        self.params = params
        activation = lambda l, k: planck_activation(l, k, params)
        deactivation = lambda l, k: constant_deactivation(l, k, params)

        super(PlanckSIS_MF, self).__init__(
            activation, deactivation, degree_dist, tol=tol, verbose=verbose, dtype=dtype
        )


class PlanckSIR_MF(ComplexSIRMF):
    def __init__(self, degree_dist, params, tol=1e-3, verbose=1, dtype="float"):
        self.params = params
        activation = lambda l, k: planck_activation(l, k, params)
        deactivation = lambda l, k: constant_deactivation(l, k, params)

        super(PlanckSIR_MF, self).__init__(
            activation, deactivation, degree_dist, tol=tol, verbose=verbose, dtype=dtype
        )
