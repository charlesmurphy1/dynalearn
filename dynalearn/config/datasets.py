from .config import Config


class DatasetConfig(Config):
    @classmethod
    def plain_markov_default(cls):
        cls = cls()
        cls.name = "Dataset"
        cls.bias = 0
        cls.replace = True
        cls.use_groundtruth = False
        return cls

    @classmethod
    def degree_weighted_markov_default(cls):
        cls = cls()
        cls.name = "DegreeWeightedMarkovDataset"
        cls.bias = 0.5
        cls.replace = True
        cls.use_groundtruth = False
        return cls

    @classmethod
    def state_weighted_markov_default(cls):
        cls = cls()
        cls.name = "StateWeightedMarkovDataset"
        cls.bias = 0.5
        cls.replace = True
        cls.use_groundtruth = False
        return cls

    @classmethod
    def state_weighted_markov_hidden_sissis_default(cls):
        cls = cls()
        cls.name = "StateWeightedMarkovDataset"
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
