from .meanfield import Meanfield
from dynalearn.dynamics.stochastic_epidemics import SIS, SIR
import numpy as np


class SISMeanfield(Meanfield):
    def __init__(self, p_k, model):
        Meanfield.__init__(self, p_k, 2, with_numba=False)
        if model.__class__ is not SIS:
            raise ValueError(
                f"{model.__class__} is invalid, the model class must be 'SIS'."
            )
        self.model = model
        self.infection = model.infection
        self.recovery = model.recovery

    def marginal_ltp(self, x, y, k, phi):

        if x == 0:
            if y == 0:
                return (1 - phi[1] * self.infection) ** k
            elif y == 1:
                return 1 - (1 - phi[1] * self.infection) ** k
        elif x == 1:
            if y == 0:
                return self.recovery
            elif y == 1:
                return 1 - self.recovery
        raise ValueError(f"{x} for x is invalid. Entries must be [0, 1]")


class SIRMeanfield(Meanfield):
    def __init__(self, p_k, model):
        Meanfield.__init__(self, p_k, 3, with_numba=False)
        if model.__class__ is not SIR:
            raise ValueError(
                f"{model.__class__} is invalid, the model class must be 'SIR'."
            )
        self.model = model
        self.infection = model.infection
        self.recovery = model.recovery

    def marginal_ltp(self, x, y, k, phi):
        if x == 0:
            if y == 0:
                return (1 - phi[1] * self.infection) ** k
            elif y == 1:
                return 1 - (1 - phi[1] * self.infection) ** k
            elif y == 2:
                return 0
            raise ValueError(f"{y} for y is invalid. Entries must be [0, 1]")
        elif x == 1:
            if y == 0:
                return 0
            elif y == 1:
                return 1 - self.recovery
            elif y == 2:
                return self.recovery
            raise ValueError(f"{y} for y is invalid. Entries must be [0, 1]")
        elif x == 2:
            if y == 0:
                return 0
            elif y == 1:
                return 0
            elif y == 2:
                return 1
            raise ValueError(f"{y} for y is invalid. Entries must be [0, 1]")

        raise ValueError(f"{x} for x is invalid. Entries must be [0, 1]")
