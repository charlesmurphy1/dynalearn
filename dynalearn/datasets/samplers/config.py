class SamplerConfig:
    @classmethod
    def BiasedSamplerDefault(cls):

        cls = cls()

        cls.sampling_bias = 0.6
        cls.replace = True
        cls.sample_from_weights = False
        cls.resample = 1000

        return cls

    @classmethod
    def RandomSamplerDefault(cls):

        cls = cls()

        cls.replace = True
        cls.sample_from_weights = False
        cls.resample = 1000

        return cls

    @classmethod
    def SequentialSamplerDefault(cls):

        cls = cls()

        cls.sample_from_weights = False
        cls.resample = 1000

        return cls
