from .dynamics import *
from .epidemics import *
from .simple_contagion import *
from .complex_contagion import *
from .interacting_contagion import *


dynamics_models = {
    "SIS": SIS,
    "SIR": SIR,
    "SoftThresholdSIS": SoftThresholdSIS,
    "SoftThresholdSIR": SoftThresholdSIR,
    "NonLinearSIS": NonLinearSIS,
    "NonLinearSIR": NonLinearSIR,
    "SineSIS": SineSIS,
    "SineSIR": SineSIR,
    "PlanckSIS": PlanckSIS,
    "PlanckSIR": PlanckSIR,
    "SISSIS": SISSIS,
}


def get(params_dict):
    name = params_dict["name"]
    params = params_dict["params"]
    if "init" not in params or params["init"] == "None":
        params["init"] = None

    if name in dynamics_models:
        return dynamics_models[name](params)
    else:
        raise ValueError(
            "Wrong name of dynamics. Valid entries are: {0}".format(
                list(dynamics_models.keys())
            )
        )
