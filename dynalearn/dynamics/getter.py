import dynalearn as dl


def get(params_dict):
    name = params_dict["name"]
    params = params_dict["params"]
    if "init_state" not in params or params["init_state"] == "None":
        params["init_state"] = None
    if name == "SIS":
        return dl.dynamics.SIS(
            params["infection_prob"], params["recovery_prob"], params["init_state"]
        )
    elif name == "SIR":
        return dl.dynamics.SIR(
            params["infection_prob"], params["recovery_prob"], params["init_state"]
        )
    elif name == "SoftThresholdSIS":
        return dl.dynamics.SoftThresholdSIS(
            params["mu"], params["beta"], params["recovery_prob"], params["init_state"]
        )
    elif name == "SoftThresholdSIR":
        return dl.dynamics.SoftThresholdSIR(
            params["mu"], params["beta"], params["recovery_prob"], params["init_state"]
        )
    elif name == "NonLinearSIS":
        return dl.dynamics.NonLinearSIS(
            params["infection_prob"],
            params["recovery_prob"],
            params["alpha"],
            params["init_state"],
        )
    elif name == "NonLinearSIR":
        return dl.dynamics.NonLinearSIR(
            params["infection_prob"],
            params["recovery_prob"],
            params["alpha"],
            params["init_state"],
        )
    elif name == "SineSIS":
        return dl.dynamics.SineSIS(
            params["infection_prob"],
            params["recovery_prob"],
            params["epsilon"],
            params["init_state"],
        )
    elif name == "SineSIR":
        return dl.dynamics.SineSIR(
            params["infection_prob"],
            params["recovery_prob"],
            params["epsilon"],
            params["init_state"],
        )
    elif name == "PlanckSIS":
        return dl.dynamics.PlanckSIS(
            params["recovery_prob"], params["temperature"], params["init_state"]
        )
    elif name == "PlanckSIR":
        return dl.dynamics.PlanckSIR(
            params["recovery_prob"], params["temperature"], params["init_state"]
        )
    else:
        raise ValueError("Wrong name of dynamics.")
