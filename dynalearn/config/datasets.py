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
    def degree_weighted_continuous_default(cls):
        cls = cls()
        cls.name = "DegreeWeightedContinuousDataset"
        cls.bias = 0.5
        cls.replace = True
        cls.use_groundtruth = False
        return cls

    @classmethod
    def strength_weighted_Discrete_default(cls):
        cls = cls()
        cls.name = "StrengthWeightedDiscreteDataset"
        cls.bias = 0.5
        cls.replace = True
        cls.use_groundtruth = False
        cls.max_num_points = 1000
        return cls

    @classmethod
    def strength_weighted_continuous_default(cls):
        cls = cls()
        cls.name = "StrengthWeightedContinuousDataset"
        cls.bias = 0.5
        cls.replace = True
        cls.use_groundtruth = False
        cls.max_num_points = 1000
        return cls

    @classmethod
    def strength_weighted_continuous_default(cls):
        cls = cls()
        cls.name = "StrengthWeightedContinuousDataset"
        cls.bias = 0.5
        cls.replace = True
        cls.use_groundtruth = False
        cls.max_num_points = 1000
        return cls

    @classmethod
    def state_weighted_discrete_default(cls):
        cls = cls()
        cls.name = "StateWeightedDiscreteDataset"
        cls.bias = 0.5
        cls.replace = True
        cls.threshold_window_size = 3
        cls.use_groundtruth = False
        return cls

    @classmethod
    def state_weighted_continuous_default(cls):
        cls = cls()
        cls.name = "StateWeightedContinuousDataset"
        cls.bias = 0.5
        cls.replace = True
        cls.use_groundtruth = False
        cls.max_num_points = 1000
        return cls

    @classmethod
    def state_weighted_hidden_sissis(cls):
        cls = cls()
        cls.name = "StateWeightedDiscreteDataset"
        cls.bias = 0.5
        cls.replace = True
        cls.use_groundtruth = False
        cls.threshold_window_size = 3
        cls.transforms = TransformConfig.hidden_sissis_default()
        return cls

    @classmethod
    def state_weighted_partially_hidden_sissis(cls):
        cls = cls()
        cls.name = "StateWeightedDiscreteDataset"
        cls.bias = 0.5
        cls.replace = True
        cls.use_groundtruth = False
        cls.threshold_window_size = 3
        cls.transforms = TransformConfig.partially_hidden_sissis_default()
        return cls

    @classmethod
    def degree_weighted_hidden_sissis(cls):
        cls = cls()
        cls.name = "DegreeWeightedDiscreteDataset"
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

    @classmethod
    def partially_hidden_sissis_default(cls):
        cls = cls()
        cls.names = ["PartiallyRemapStateTransform"]
        return cls
