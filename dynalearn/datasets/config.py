class DatasetConfig(object):
    @classmethod
    def plain_default(cls):
        cls = cls()
        cls.name = "Dataset"
        cls.bias = 0
        cls.replace = True
        cls.with_truth = False
        cls.resampling_time = 2
        return cls

    @classmethod
    def degree_weighted_default(cls):
        cls = cls()
        cls.name = "DegreeWeightedDataset"
        cls.bias = 0.5
        cls.replace = True
        cls.with_truth = False
        cls.resampling_time = 2
        return cls

    @classmethod
    def state_weighted_default(cls):
        cls = cls()
        cls.name = "StateWeightedDataset"
        cls.bias = 0.5
        cls.replace = True
        cls.with_truth = True
        cls.resampling_time = 2
        return cls
