from .meanfield import *
from .complex import *
from .simple import *
from .interacting import *
from dynalearn.dynamics import *

__meanfields__ = {
    "Generic": lambda model: lambda p_k: GenericMeanfield(p_k, model),
    "SIS": lambda model: lambda p_k: SISMeanfield(p_k, model),
    "SIR": lambda model: lambda p_k: SIRMeanfield(p_k, model),
    "ThresholdSIS": lambda model: lambda p_k: ThresholdSISMeanfield(p_k, model),
    "ThresholdSIR": lambda model: lambda p_k: ThresholdSIRMeanfield(p_k, model),
    "NonLinearSIS": lambda model: lambda p_k: NonLinearSISMeanfield(p_k, model),
    "NonLinearSIR": lambda model: lambda p_k: NonLinearSIRMeanfield(p_k, model),
    "SineSIS": lambda model: lambda p_k: SineSISMeanfield(p_k, model),
    "SineSIR": lambda model: lambda p_k: SineSIRMeanfield(p_k, model),
    "PlanckSIS": lambda model: lambda p_k: PlanckSISMeanfield(p_k, model),
    "PlanckSIR": lambda model: lambda p_k: PlanckSIRMeanfield(p_k, model),
    "SISSIS": lambda model: lambda p_k: SISSISMeanfield(p_k, model),
    "HiddenSISSIS": lambda model: lambda p_k: SISSISMeanfield(p_k, model),
}


def get(model):
    name = model.__class__.__name__
    if name in __meanfields__:
        return __meanfields__[name](model)
    elif hasattr(model, "predict"):
        return __meanfields__["Generic"](model)
    else:
        raise ValueError(
            f"'{model}' is invalid, models must have 'predict' implemented."
        )
