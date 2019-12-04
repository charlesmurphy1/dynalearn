import dynalearn as dl

dynamics_models = {
    "SIS": dl.dynamics.SIS,
    "SIR": dl.dynamics.SIR,
    "SoftThresholdSIS": dl.dynamics.SoftThresholdSIS,
    "SoftThresholdSIR": dl.dynamics.SoftThresholdSIR,
    "NonLinearSIS": dl.dynamics.NonLinearSIS,
    "NonLinearSIR": dl.dynamics.NonLinearSIR,
    "SineSIS": dl.dynamics.SineSIS,
    "SineSIR": dl.dynamics.SineSIR,
    "PlanckSIS": dl.dynamics.PlanckSIS,
    "PlanckSIR": dl.dynamics.PlanckSIR,
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
