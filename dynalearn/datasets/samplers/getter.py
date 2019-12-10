from .random_sampler import *
from .sequential_sampler import *
from .biased_sampler import *

samplers = {
    "SequentialSampler": SequentialSampler,
    "RandomSampler": RandomSampler,
    "DegreeBiasedSampler": DegreeBiasedSampler,
    "StateBiasedSampler": StateBiasedSampler,
}


def get(config):
    name = config["name"]
    _config = config["config"]
    if name in samplers:
        return samplers[name]("train", _config)
    else:
        raise ValueError(
            "Wrong name of sampler. Valid entries are: {0}".format(
                list(samplers.keys())
            )
        )
