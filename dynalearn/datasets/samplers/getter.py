import dynalearn as dl

samplers = [
    "SequentialSampler",
    "RandomSampler",
    "DegreeBiasedSampler",
    "StateBiasedSampler",
]


def get(params_dict, dynamics):
    name = params_dict["name"]
    config = params_dict["config"]

    if name == "SequentialSampler":
        return dl.dynalearn.datasets.samplers.SequentialSampler("train", config)
    elif name == "RandomSampler":
        return dl.dynalearn.datasets.samplers.RandomSampler("train", config)
    elif name == "DegreeBiasedSampler":
        return dl.dynalearn.datasets.samplers.DegreeBiasedSampler("train", config)
    elif name == "StateBiasedSampler":

        return dl.dynalearn.datasets.samplers.StateBiasedSampler(
            "train", dynamics, config
        )
    else:
        raise ValueError(
            "Wrong name of sampler. Valid entries are: {0}".format(samplers)
        )
