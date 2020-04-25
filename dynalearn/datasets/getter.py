from dynalearn.datasets import Dataset, DegreeWeightedDataset, StateWeightedDataset


__datasets__ = {
    "Dataset": Dataset,
    "DegreeWeightedDataset": DegreeWeightedDataset,
    "StateWeightedDataset": StateWeightedDataset,
}


def get(config):
    name = config.name
    if name in __datasets__:
        return __datasets__[name](config)
    else:
        raise ValueError(
            f"{name} is invalid, possible entries are {list(__datasets__.keys())}"
        )
