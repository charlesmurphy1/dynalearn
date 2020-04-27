import numpy as np
from .base import GenericMeanfield
from dynalearn.dynamics.complex import *


class ThresholdSISMeanfield(GenericMeanfield):
    def __init__(self, p_k, model, with_numba=False):
        if model.__class__ is not ThresholdSIS:
            raise ValueError(
                f"{model.__class__} is invalid, the model class must be 'ThresholdSIS'."
            )
        GenericMeanfield.__init__(self, p_k, model, with_numba=with_numba)


class ThresholdSIRMeanfield(GenericMeanfield):
    def __init__(self, p_k, model, with_numba=False):
        if model.__class__ is not ThresholdSIR:
            raise ValueError(
                f"{model.__class__} is invalid, the model class must be 'ThresholdSIR'."
            )
        GenericMeanfield.__init__(self, p_k, model, with_numba=with_numba)


class NonLinearSISMeanfield(GenericMeanfield):
    def __init__(self, p_k, model, with_numba=False):
        if model.__class__ is not NonLinearSIS:
            raise ValueError(
                f"{model.__class__} is invalid, the model class must be 'NonLinearSIS'."
            )
        GenericMeanfield.__init__(self, p_k, model, with_numba=with_numba)


class NonLinearSIRMeanfield(GenericMeanfield):
    def __init__(self, p_k, model, with_numba=False):
        if model.__class__ is not NonLinearSIR:
            raise ValueError(
                f"{model.__class__} is invalid, the model class must be 'NonLinearSIR'."
            )
        GenericMeanfield.__init__(self, p_k, model, with_numba=with_numba)


class SineSISMeanfield(GenericMeanfield):
    def __init__(self, p_k, model, with_numba=False):
        if model.__class__ is not SineSIS:
            raise ValueError(
                f"{model.__class__} is invalid, the model class must be 'SineSIS'."
            )
        GenericMeanfield.__init__(self, p_k, model, with_numba=with_numba)


class SineSIRMeanfield(GenericMeanfield):
    def __init__(self, p_k, model, with_numba=False):
        if model.__class__ is not SineSIR:
            raise ValueError(
                f"{model.__class__} is invalid, the model class must be 'SineSIR'."
            )
        GenericMeanfield.__init__(self, p_k, model, with_numba=with_numba)


class PlanckSISMeanfield(GenericMeanfield):
    def __init__(self, p_k, model, with_numba=False):
        if model.__class__ is not PlanckSIS:
            raise ValueError(
                f"{model.__class__} is invalid, the model class must be 'PlanckSIS'."
            )
        GenericMeanfield.__init__(self, p_k, model, with_numba=with_numba)


class PlanckSIRMeanfield(GenericMeanfield):
    def __init__(self, p_k, model, with_numba=False):
        if model.__class__ is not PlanckSIR:
            raise ValueError(
                f"{model.__class__} is invalid, the model class must be 'PlanckSIR'."
            )
        GenericMeanfield.__init__(self, p_k, model, with_numba=with_numba)
