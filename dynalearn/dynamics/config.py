from dynalearn.nn.optimizer import *
from dynalearn.utilities import Config


class DynamicsConfig(Config):
    @classmethod
    def sis_default(cls):
        cls = cls()
        cls.name = "SIS"
        cls.infection = 0.04
        cls.recovery = 0.08
        cls.initial_infected = -1
        return cls

    @classmethod
    def plancksis_default(cls):
        cls = cls()
        cls.name = "PlanckSIS"
        cls.temperature = 6.0
        cls.recovery = 0.08
        cls.initial_infected = -1
        return cls

    @classmethod
    def sissis_default(cls):
        cls = cls()
        cls.name = "SISSIS"
        cls.infection1 = 0.02
        cls.infection2 = 0.01
        cls.recovery1 = 0.12
        cls.recovery2 = 0.13
        cls.coupling = 10.0
        cls.initial_infected = -1
        return cls

    @classmethod
    def gnn_test(cls):
        cls = cls()
        cls.name = "LearnableEpidemics"
        cls.num_states = 2

        cls.loss = "weighted_cross_entropy"
        cls.optimizer = OptimizerConfig.radam_default()

        cls.in_activation = "relu"
        cls.att_activation = "relu"
        cls.out_activation = "relu"

        cls.in_channels = [2]
        cls.att_channels = 2
        cls.heads = 1
        cls.out_channels = [2]
        cls.concat = False
        cls.bias = True

        return cls

    @classmethod
    def sis_gnn_default(cls):
        cls = cls()
        cls.name = "LearnableEpidemics"
        cls.num_states = 2

        cls.loss = "weighted_cross_entropy"
        cls.optimizer = OptimizerConfig.radam_default()

        cls.in_activation = "relu"
        cls.att_activation = "relu"
        cls.out_activation = "relu"

        cls.in_channels = [32]
        cls.att_channels = 32
        cls.heads = 1
        cls.out_channels = [32]
        cls.concat = True
        cls.bias = True

        return cls

    @classmethod
    def plancksis_gnn_default(cls):
        cls = cls()
        cls.name = "LearnableEpidemics"
        cls.num_states = 2

        cls.loss = "weighted_cross_entropy"
        cls.optimizer = OptimizerConfig.radam_default()

        cls.in_activation = "relu"
        cls.att_activation = "relu"
        cls.out_activation = "relu"

        cls.in_channels = [32]
        cls.att_channels = 32
        cls.heads = 2
        cls.out_channels = [32]
        cls.concat = True
        cls.bias = True

        return cls

    @classmethod
    def sissis_gnn_default(cls):
        cls = cls()
        cls.name = "LearnableEpidemics"
        cls.num_states = 4

        cls.loss = "weighted_cross_entropy"
        cls.optimizer = OptimizerConfig.radam_default()

        cls.in_activation = "relu"
        cls.att_activation = "relu"
        cls.out_activation = "relu"

        cls.in_channels = [32, 32]
        cls.att_channels = 32
        cls.heads = 2
        cls.out_channels = [32, 32]
        cls.concat = True
        cls.bias = True

        return cls
