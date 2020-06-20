from dynalearn.datasets import (
    MarkovDataset,
    DegreeWeightedMarkovDataset,
    StateWeightedMarkovDataset,
    GeneralMarkovDataset,
    DegreeWeightedGeneralMarkovDataset,
    StateWeightedGeneralMarkovDataset,
)


__datasets__ = {
    "MarkovDataset": MarkovDataset,
    "DegreeWeightedMarkovDataset": DegreeWeightedMarkovDataset,
    "StateWeightedMarkovDataset": StateWeightedMarkovDataset,
    "GeneralMarkovDataset": GeneralMarkovDataset,
    "DegreeWeightedGeneralMarkovDataset": DegreeWeightedGeneralMarkovDataset,
    "StateWeightedGeneralMarkovDataset": StateWeightedGeneralMarkovDataset,
}


def get(config):
    name = config.name
    if name in __datasets__:
        return __datasets__[name](config)
    else:
        raise ValueError(
            f"{name} is invalid, possible entries are {list(__datasets__.keys())}"
        )
