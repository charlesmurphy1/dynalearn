import dynalearn.dynamics as dn


class SamplerConfig:
    @classmethod
    def BiasedSamplerDefault(cls, dynamics_config):

        cls = cls()

        dynamics = dn.get(dynamics_config)
        cls.dynamics_states = list(dynamics.state_label.values())
        cls.sampling_bias = 0.6
        cls.replace = True
        cls.sample_from_weights = True
        cls.resample = 1000

        return cls

    @classmethod
    def RandomSamplerDefault(cls):

        cls = cls()

        cls.replace = True
        cls.sample_from_weights = True
        cls.resample = 1000

        return cls

    @classmethod
    def SequentialSamplerDefault(cls):

        cls = cls()

        cls.sample_from_weights = True
        cls.resample = 1000

        return cls
