from .epidemics import (
    SIS,
    SIR,
    ThresholdSIS,
    ThresholdSIR,
    NonLinearSIS,
    NonLinearSIR,
    SineSIS,
    SineSIR,
    PlanckSIS,
    PlanckSIR,
    SISSIS,
    HiddenSISSIS,
    PartiallyHiddenSISSIS,
    TrainableEpidemics,
)
from .metapopulation import MetaSIS, TrainableMetapopulation


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
    "PartiallyHiddenSISSIS": PartiallyHiddenSISSIS,
    "TrainableEpidemics": TrainableEpidemics,
    "MetaSIS": MetaSIS,
    "TrainableMetapopulation": TrainableMetapopulation,
}


def get(config):
    name = config.name
    if name in __dynamics__:
        return __dynamics__[name](config)
    else:
        raise ValueError(
            f"{name} is invalid, possible entries are {list(__dynamics__.keys())}"
        )
