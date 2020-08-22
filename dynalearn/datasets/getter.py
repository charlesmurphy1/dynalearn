from dynalearn.datasets import (
    DiscreteDataset,
    DegreeWeightedDiscreteDataset,
    StrengthWeightedDiscreteDataset,
    StateWeightedDiscreteDataset,
    ContinuousDataset,
    DegreeWeightedContinuousDataset,
    StrengthWeightedContinuousDataset,
    StateWeightedContinuousDataset,
)


__datasets__ = {
    "DiscreteDataset": DiscreteDataset,
    "DegreeWeightedDiscreteDataset": DegreeWeightedDiscreteDataset,
    "StrengthWeightedDiscreteDataset": StrengthWeightedDiscreteDataset,
    "StateWeightedDiscreteDataset": StateWeightedDiscreteDataset,
    "ContinuousDataset": ContinuousDataset,
    "DegreeWeightedContinuousDataset": DegreeWeightedContinuousDataset,
    "StrengthWeightedContinuousDataset": StrengthWeightedContinuousDataset,
    "StateWeightedContinuousDataset": StateWeightedContinuousDataset,
}


def get(config):
    name = config.name
    if name in __datasets__:
        return __datasets__[name](config)
    else:
        raise ValueError(
            f"{name} is invalid, possible entries are {list(__datasets__.keys())}"
        )
