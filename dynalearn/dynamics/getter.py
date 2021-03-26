from .stochastic_epidemics import (
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
    AsymmetricSISSIS,
    HiddenSISSIS,
    PartiallyHiddenSISSIS,
)
from .deterministic_epidemics import DSIS, DSIR
from .reaction_diffusion import RDSIS, RDSIR
from .trainable import (
    GNNSEDynamics,
    GNNDEDynamics,
    TrainableReactionDiffusion,
)

__dynamics__ = {
    "SIS": SIS,
    "SIR": SIR,
    "DSIS": DSIS,
    "DSIR": DSIR,
    "ThresholdSIR": ThresholdSIR,
    "ThresholdSIS": ThresholdSIS,
    "NonLinearSIS": NonLinearSIS,
    "NonLinearSIR": NonLinearSIR,
    "SineSIS": SineSIS,
    "SineSIR": SineSIR,
    "PlanckSIS": PlanckSIS,
    "PlanckSIR": PlanckSIR,
    "SISSIS": SISSIS,
    "AsymmetricSISSIS": AsymmetricSISSIS,
    "HiddenSISSIS": HiddenSISSIS,
    "PartiallyHiddenSISSIS": PartiallyHiddenSISSIS,
    "RDSIS": RDSIS,
    "RDSIR": RDSIR,
    "GNNSEDynamics": GNNSEDynamics,
    "GNNDEDynamics": GNNDEDynamics,
    "TrainableReactionDiffusion": TrainableReactionDiffusion,
}


def get(config):
    name = config.name
    if name in __dynamics__:
        return __dynamics__[name](config)
    else:
        raise ValueError(
            f"{name} is invalid, possible entries are {list(__dynamics__.keys())}"
        )
