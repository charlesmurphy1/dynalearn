from .config import Config


class DiscreteDatasetConfig(Config):
    @classmethod
    def plain(cls):
        cls = cls()
        cls.name = "DiscreteDataset"
        cls.bias = 0
        cls.replace = True
        cls.use_groundtruth = False
        return cls

    @classmethod
    def structure(cls, use_strength=True):
        cls = cls()
        cls.name = "DiscreteStructureWeightDataset"
        cls.bias = 0
        cls.replace = True
        cls.use_groundtruth = False
        cls.use_strength = use_strength
        return cls

    @classmethod
    def state(cls, use_strength=True, compounded=True):
        cls = cls()
        cls.name = "DiscreteStateWeightDataset"
        cls.bias = 0
        cls.replace = True
        cls.use_groundtruth = False
        cls.use_strength = use_strength
        cls.compounded = compounded
        return cls

    @classmethod
    def hidden_sissis(cls):
        cls = cls()
        cls.name = "DiscreteStateWeightDataset"
        cls.bias = 0
        cls.replace = True
        cls.use_groundtruth = False
        cls.use_strength = False
        cls.compounded = True
        cls.transforms = TransformConfig.hidden_sissis_default()

        return cls

    @classmethod
    def partially_hidden_sissis(cls):
        cls = cls()
        cls.name = "DiscreteStateWeightDataset"
        cls.bias = 0
        cls.replace = True
        cls.use_groundtruth = False
        cls.use_strength = False
        cls.compounded = True
        cls.transforms = TransformConfig.partially_hidden_sissis_default()

        return cls


class ContinuousDatasetConfig(Config):
    @classmethod
    def plain(cls):
        cls = cls()
        cls.name = "ContinuousDataset"
        cls.bias = 0
        cls.replace = True
        cls.use_groundtruth = False
        return cls

    @classmethod
    def structure(cls, use_strength=True):
        cls = cls()
        cls.name = "ContinuousStructureWeightDataset"
        cls.bias = 0
        cls.replace = True
        cls.use_groundtruth = False
        cls.use_strength = use_strength
        return cls

    @classmethod
    def state(cls, use_strength=True, compounded=False, reduce=False, total=True):
        cls = cls()
        cls.name = "ContinuousStateWeightDataset"
        cls.bias = 0
        cls.replace = True
        cls.use_groundtruth = False
        cls.use_strength = use_strength
        cls.compounded = compounded
        cls.total = total
        cls.reduce = reduce
        cls.max_num_points = -1
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
