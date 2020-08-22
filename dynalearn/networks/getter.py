from .generative import *
from .weighted import *
from .real_network import *


__networks__ = {
    "ERNetwork": ERNetwork,
    "BANetwork": BANetwork,
    "ConfigurationNetwork": ConfigurationNetwork,
    "RealNetwork": RealNetwork,
    "RealTemporalNetwork": RealTemporalNetwork,
}

__weights__ = {
    "UniformWeightGenerator": UniformWeightGenerator,
    "LogUniformWeightGenerator": LogUniformWeightGenerator,
    "NormalWeightGenerator": NormalWeightGenerator,
    "LogNormalWeightGenerator": LogNormalWeightGenerator,
    "DegreeWeightGenerator": DegreeWeightGenerator,
    "BetweennessWeightGenerator": BetweennessWeightGenerator,
}


def get(config):
    name = config.name
    if "weights" in config.__dict__:
        if config.weights.name in __weights__:
            weight_gen = __weights__[config.weights.name](config.weights)
        else:
            raise ValueError(
                f"{config.weights.name} is invalid, possible entries are {list(__weights__.keys())}"
            )
    else:
        weight_gen = None
    if name in __networks__:
        return __networks__[name](config, weight_gen)
    else:
        raise ValueError(
            f"{name} is invalid, possible entries are {list(__networks__.keys())}"
        )
