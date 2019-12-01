from dynalearn.dynamics import *


def get(params_dict):
    name = params_dict["name"]
    params = params_dict["params"]
    if name == "SIS":
        return SIS(
            params["infection_prob"], params["recovery_prob"], params["init_state"]
        )
    elif name == "SIR":
        return SIR(
            params["infection_prob"], params["recovery_prob"], params["init_state"]
        )
    elif name == "SoftThresholdSIS":
        return SoftThresholdSIS(params["mu"], params["beta"], params["recovery_prob"])
    elif name == "SoftThresholdSIR":
        return SoftThresholdSIR(params["mu"], params["beta"], params["recovery_prob"])
    elif name == "NonLinearSIS":
        return NonLinearSIS(
            params["infection_prob"],
            params["recovery_prob"],
            params["alpha"],
            params["init_state"],
        )
    elif name == "NonLinearSIR":
        return NonLinearSIR(
            params["infection_prob"],
            params["recovery_prob"],
            params["alpha"],
            params["init_state"],
        )
    elif name == "SineSIS":
        return SineSIS(
            params["infection_prob"],
            params["recovery_prob"],
            params["epsilon"],
            params["init_state"],
        )
    elif name == "SineSIR":
        return SineSIR(
            params["infection_prob"],
            params["recovery_prob"],
            params["epsilon"],
            params["init_state"],
        )
    elif name == "PlanckSIS":
        return PlanckSIS(
            params["recovery_prob"], params["temperature"], params["init_state"]
        )
    elif name == "PlanckSIR":
        return PlanckSIR(
            params["recovery_prob"], params["temperature"], params["init_state"]
        )
    else:
        raise ValueError("Wrong name of dynamics.")
