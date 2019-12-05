from .simple_contagion import *
from .complex_contagion import *
from .interacting_contagion import *
from .gnn import *


meanfields = {
    "SIS": SIS_MF,
    "SIR": SIR_MF,
    "SoftThresholdSIS": SoftThresholdSIS_MF,
    "SoftThresholdSIR": SoftThresholdSIR_MF,
    "NonLinearSIS": NonLinearSIS_MF,
    "NonLinearSIR": NonLinearSIR_MF,
    "SineSIS": SineSIS_MF,
    "SineSIR": SineSIR_MF,
    "PlanckSIS": PlanckSIS_MF,
    "PlanckSIR": PlanckSIR_MF,
    "SISSIS": SISSIS_MF,
}


def get(dynamics, degree_dist):
    name = type(dynamics).__name__
    if name in meanfields:
        return meanfields[name](degree_dist, dynamics.params)
    else:
        raise ValueError(
            "Wrong name of meanfields. Valid entries are: {0}".format(
                list(meanfields.keys())
            )
        )
