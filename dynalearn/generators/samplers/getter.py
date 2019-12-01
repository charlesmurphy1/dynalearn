from dynalearn.generators.samplers import *
from dynalearn.dynamics import get_dynamics


def get(dynamics, params_dict):
    name = params_dict["name"]
    params = params_dict["params"]

    if name == "SequentialSampler":
        return SequentialSampler(
            "train",
            sample_from_weights=params["sample_from_weights"],
            resample=params["resample"],
        )
    elif name == "RandomSampler":
        return RandomSampler(
            "train",
            replace=params["replace"],
            sample_from_weights=params["sample_from_weights"],
            resample=params["resample"],
        )
    elif name == "DegreeBiasedSampler":
        return DegreeBiasedSampler(
            "train",
            sampling_bias=params["sampling_bias"],
            replace=params["replace"],
            sample_from_weights=params["sample_from_weights"],
            resample=params["resample"],
        )
    elif name == "StateBiasedSampler":

        return StateBiasedSampler(
            "train",
            dynamics,
            sampling_bias=params["sampling_bias"],
            replace=params["replace"],
            sample_from_weights=params["sample_from_weights"],
            resample=params["resample"],
        )
    else:
        raise ValueError("Wrong name of sampler.")
