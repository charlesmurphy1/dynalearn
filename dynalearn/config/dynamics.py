from dynalearn.nn.optimizer import *
from .config import Config
from .optimizers import OptimizerConfig


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
    def asymmetric_sissis_default(cls):
        cls = cls()
        cls.name = "SISSIS"
        cls.infection1 = 0.02
        cls.infection2 = 0.01
        cls.recovery1 = 0.12
        cls.recovery2 = 0.13
        cls.coupling = 10.0
        cls.boost = "source"
        cls.initial_infected = -1
        return cls

    @classmethod
    def hidden_sissis_default(cls):
        cls = cls()
        cls.name = "HiddenSISSIS"
        cls.infection1 = 0.02
        cls.infection2 = 0.01
        cls.recovery1 = 0.12
        cls.recovery2 = 0.13
        cls.coupling = 10.0
        cls.initial_infected = -1
        return cls

    @classmethod
    def partially_hidden_sissis_default(cls):
        cls = cls()
        cls.name = "PartiallyHiddenSISSIS"
        cls.infection1 = 0.02
        cls.infection2 = 0.01
        cls.recovery1 = 0.12
        cls.recovery2 = 0.13
        cls.coupling = 10.0
        cls.hide_prob = 0.0
        cls.initial_infected = -1
        return cls

    @classmethod
    def metasis_default(cls):
        cls = cls()
        cls.name = "MetaSIS"
        cls.infection_prob = 0.04
        cls.recovery_prob = 0.08
        cls.infection_type = 2
        cls.diffusion_susceptible = 0.1
        cls.diffusion_infected = 0.1
        cls.initial_density = 1000
        cls.initial_state_dist = -1
        return cls

    @classmethod
    def gnn_test(cls):
        cls = cls()
        cls.name = "TrainableEpidemics"
        cls.num_states = 2
        cls.window_size = 1
        cls.window_step = 1
        cls.with_non_edge = False

        cls.loss = "weighted_cross_entropy"
        cls.optimizer = OptimizerConfig.radam_default()

        cls.in_activation = "relu"
        cls.att_activation = "relu"
        cls.out_activation = "relu"

        cls.edge_activation = "relu"
        cls.edge_att_activation = "relu"

        cls.in_channels = [2]
        cls.att_channels = 2
        cls.out_channels = [2]
        cls.edge_channels = 1
        cls.edge_att_channels = 1
        cls.heads = 1
        cls.concat = False
        cls.bias = True
        cls.attn_bias = True
        cls.self_attention = True

        return cls

    @classmethod
    def sis_gnn_default(cls):
        cls = cls()
        cls.name = "TrainableEpidemics"
        cls.gnn_layer_name = "DynamicsGAT"
        cls.num_states = 2
        cls.window_size = 1
        cls.window_step = 1
        cls.with_non_edge = False

        cls.loss = "weighted_cross_entropy"
        cls.optimizer = OptimizerConfig.radam_default()

        cls.in_activation = "relu"
        cls.att_activation = "relu"
        cls.out_activation = "relu"

        cls.edge_activation = "relu"
        cls.edge_att_activation = "relu"

        cls.in_channels = [32]
        cls.att_channels = 32
        cls.out_channels = [32]
        cls.edge_channels = [4]
        cls.edge_att_channels = 4
        cls.heads = 1
        cls.concat = True
        cls.bias = True
        cls.attn_bias = True
        cls.self_attention = True

        return cls

    @classmethod
    def plancksis_gnn_default(cls):
        cls = cls()
        cls.name = "TrainableEpidemics"
        cls.gnn_layer_name = "DynamicsGAT"
        cls.num_states = 2
        cls.window_size = 1
        cls.window_step = 1
        cls.with_non_edge = False

        cls.loss = "weighted_cross_entropy"
        cls.optimizer = OptimizerConfig.radam_default()

        cls.in_activation = "relu"
        cls.att_activation = "relu"
        cls.out_activation = "relu"

        cls.edge_activation = "relu"
        cls.edge_att_activation = "relu"

        cls.in_channels = [32]
        cls.att_channels = 32
        cls.out_channels = [32]
        cls.edge_channels = [4]
        cls.edge_att_channels = 4
        cls.heads = 1
        cls.concat = True
        cls.bias = True
        cls.attn_bias = True
        cls.self_attention = True

        return cls

    @classmethod
    def sissis_gnn_default(cls):
        cls = cls()
        cls.name = "TrainableEpidemics"
        cls.gnn_layer_name = "DynamicsGAT"
        cls.num_states = 4
        cls.window_size = 1
        cls.window_step = 1
        cls.with_non_edge = False

        cls.loss = "weighted_cross_entropy"
        cls.optimizer = OptimizerConfig.radam_default()

        cls.in_activation = "relu"
        cls.att_activation = "relu"
        cls.out_activation = "relu"

        cls.edge_activation = "relu"
        cls.edge_att_activation = "relu"

        cls.in_channels = [32, 32]
        cls.att_channels = 32
        cls.out_channels = [32, 32]
        cls.edge_channels = [4]
        cls.edge_att_channels = 4
        cls.heads = 2
        cls.concat = True
        cls.bias = True
        cls.attn_bias = True
        cls.self_attention = True

        return cls

    @classmethod
    def hidden_sissis_gnn_default(cls):
        cls = cls()
        cls.name = "TrainableEpidemics"
        cls.gnn_layer_name = "DynamicsGAT"
        cls.num_states = 2
        cls.window_size = 1
        cls.window_step = 1
        cls.with_non_edge = True

        cls.loss = "weighted_cross_entropy"
        cls.optimizer = OptimizerConfig.radam_default()

        cls.in_activation = "relu"
        cls.att_activation = "relu"
        cls.out_activation = "relu"

        cls.edge_activation = "relu"
        cls.edge_att_activation = "relu"

        cls.in_channels = [32, 32]
        cls.att_channels = 32
        cls.out_channels = [32, 32]
        cls.edge_channels = [4]
        cls.edge_att_channels = 4
        cls.heads = 2
        cls.concat = True
        cls.bias = True
        cls.attn_bias = True
        cls.self_attention = True
        cls.with_non_edge = False

        return cls

    @classmethod
    def partially_hidden_sissis_gnn_default(cls):
        cls = cls()
        cls.name = "TrainableEpidemics"
        cls.gnn_layer_name = "DynamicsGAT"
        cls.num_states = 4
        cls.window_size = 1
        cls.window_step = 1
        cls.with_non_edge = True

        cls.loss = "weighted_cross_entropy"
        cls.optimizer = OptimizerConfig.radam_default()

        cls.in_activation = "relu"
        cls.att_activation = "relu"
        cls.out_activation = "relu"

        cls.edge_activation = "relu"
        cls.edge_att_activation = "relu"

        cls.in_channels = [32, 32]
        cls.att_channels = 32
        cls.out_channels = [32, 32]
        cls.edge_channels = [4]
        cls.edge_att_channels = 4
        cls.heads = 2
        cls.concat = True
        cls.bias = True
        cls.attn_bias = True
        cls.self_attention = True
        cls.with_non_edge = False

        return cls
