from .generative import *
from .real_network import *


__networks__ = {
    "ERNetwork": ERNetwork,
    "BANetwork": BANetwork,
    "ConfigurationNetwork": ConfigurationNetwork,
    "RealNetwork": RealNetwork,
    "RealTemporalNetwork": RealTemporalNetwork,
}


def get(config):
    name = config.name
    if name in __networks__:
        return __networks__[name](config)
    else:
        raise ValueError(
            f"{name} is invalid, possible entries are {list(__networks__.keys())}"
        )
