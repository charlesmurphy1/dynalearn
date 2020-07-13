from .config import Config


class DatasetConfig(Config):
    @classmethod
    def plain_discrete_default(cls):
        cls = cls()
        cls.name = "DiscreteDataset"
        cls.bias = 0
        cls.replace = True
        cls.use_groundtruth = False
        return cls

    @classmethod
    def degree_weighted_discrete_default(cls):
        cls = cls()
        cls.name = "DegreeWeightedDiscreteDataset"
        cls.bias = 0.5
        cls.replace = True
        cls.use_groundtruth = False
        return cls

    @classmethod
    def state_weighted_discrete_default(cls):
        cls = cls()
        cls.name = "StateWeightedDiscreteDataset"
        cls.bias = 0.5
        cls.replace = True
        cls.use_groundtruth = False
        return cls

    @classmethod
    def state_weighted_hidden_sissis(cls):
        cls = cls()
        cls.name = "StateWeightedDiscreteDataset"
        cls.bias = 0.5
        cls.replace = True
        cls.use_groundtruth = False
        cls.transforms = TransformConfig.hidden_sissis_default()
        return cls


class TransformConfig(Config):
    @classmethod
    def hidden_sissis_default(cls):
        cls = cls()
        cls.names = ["RemapStateTransform"]
        return cls
