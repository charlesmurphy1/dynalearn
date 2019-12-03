import dynalearn as dl


def get(params_dict, dynamics):
    name = params_dict["name"]
    params = params_dict["params"]

    if name == "SequentialSampler":
        return dl.dynalearn.generators.samplers.SequentialSampler(
            "train",
            sample_from_weights=params["sample_from_weights"],
            resample=params["resample"],
        )
    elif name == "RandomSampler":
        return dl.dynalearn.generators.samplers.RandomSampler(
            "train",
            replace=params["replace"],
            sample_from_weights=params["sample_from_weights"],
            resample=params["resample"],
        )
    elif name == "DegreeBiasedSampler":
        return dl.dynalearn.generators.samplers.DegreeBiasedSampler(
            "train",
            sampling_bias=params["sampling_bias"],
            replace=params["replace"],
            sample_from_weights=params["sample_from_weights"],
            resample=params["resample"],
        )
    elif name == "StateBiasedSampler":

        return dl.dynalearn.generators.samplers.StateBiasedSampler(
            "train",
            dynamics,
            sampling_bias=params["sampling_bias"],
            replace=params["replace"],
            sample_from_weights=params["sample_from_weights"],
            resample=params["resample"],
        )
    else:
        raise ValueError("Wrong name of sampler.")
