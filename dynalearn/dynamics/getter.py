from .simple import *
from .complex import *
from .interacting import *
from .trainable import *


__dynamics__ = {
    "SIS": SIS,
    "SIR": SIR,
    "ThresholdSIS": ThresholdSIS,
    "ThresholdSIR": ThresholdSIR,
    "NonLinearSIS": NonLinearSIS,
    "NonLinearSIR": NonLinearSIR,
    "SineSIS": SineSIS,
    "SineSIR": SineSIR,
    "PlanckSIS": PlanckSIS,
    "PlanckSIR": PlanckSIR,
    "SISSIS": SISSIS,
    "HiddenSISSIS": HiddenSISSIS,
    "TrainableEpidemics": TrainableEpidemics,
}


def get(config):
    name = config.name
    if name in __dynamics__:
        return __dynamics__[name](config)
    else:
        raise ValueError(
            f"{name} is invalid, possible entries are {list(__dynamics__.keys())}"
        )
