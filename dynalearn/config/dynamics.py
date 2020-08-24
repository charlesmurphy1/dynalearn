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
        cls.density = 100
        cls.state_dist = -1
        return cls

    @classmethod
    def metasir_default(cls):
        cls = cls()
        cls.name = "MetaSIR"
        # cls.infection_prob = 2.5 / 2.3
        # cls.recovery_prob = 1.0 / 7.5
        cls.infection_prob = 0.04
        cls.recovery_prob = 0.08
        cls.infection_type = 2
        cls.diffusion_susceptible = 0.1
        cls.diffusion_infected = 0.1
        cls.diffusion_recovered = 0.1
        cls.density = 100
        cls.state_dist = -1
        return cls

    @classmethod
    def metasir_covid(cls):
        cls = cls()
        cls.name = "MetaSIR"
        cls.infection_prob = 2.5 / 2.3
        cls.recovery_prob = 1.0 / (7.5)
        cls.infection_type = 2
        cls.diffusion_susceptible = 0.1
        cls.diffusion_infected = 0.1
        cls.diffusion_recovered = 0.1
        cls.density = array(
            [
                331549.0,
                388167.0,
                1858683.0,
                716820.0,
                157640.0,
                673559.0,
                1149460.0,
                5664579.0,
                356958.0,
                394151.0,
                1240155.0,
                579962.0,
                495761.0,
                782979.0,
                1119596.0,
                196329.0,
                771044.0,
                914678.0,
                257762.0,
                723576.0,
                521870.0,
                220461.0,
                633564.0,
                460001.0,
                434930.0,
                316798.0,
                329587.0,
                6663394.0,
                1661785.0,
                1493898.0,
                654214.0,
                307651.0,
                1022800.0,
                160980.0,
                1120406.0,
                942665.0,
                330119.0,
                1032983.0,
                581078.0,
                153129.0,
                1942389.0,
                88636.0,
                804664.0,
                134137.0,
                694844.0,
                2565124.0,
                519546.0,
                1152651.0,
                172539.0,
                964693.0,
                84777.0,
                86487.0,
            ]
        )
        cls.state_dist = -1
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
        cls.gnn_activation = "relu"
        cls.out_activation = "relu"

        cls.edge_activation = "relu"
        cls.edge_gnn_activation = "relu"

        cls.in_channels = [2]
        cls.gnn_channels = 2
        cls.out_channels = [2]
        cls.edge_channels = 1
        cls.edge_gnn_channels = 1
        cls.heads = 1
        cls.concat = False
        cls.bias = True
        cls.self_attention = True

        return cls

    @classmethod
    def sis_gnn_default(cls):
        cls = cls()
        cls.name = "TrainableEpidemics"
        cls.gnn_name = "DynamicsGATConv"
        cls.num_states = 2
        cls.window_size = 1
        cls.window_step = 1
        cls.with_non_edge = False

        cls.loss = "weighted_cross_entropy"
        cls.optimizer = OptimizerConfig.radam_default()

        cls.in_activation = "relu"
        cls.gnn_activation = "relu"
        cls.out_activation = "relu"

        cls.edge_activation = "relu"
        cls.edge_gnn_activation = "relu"

        cls.in_channels = [32]
        cls.gnn_channels = 32
        cls.out_channels = [32]
        cls.edge_channels = [4]
        cls.edge_gnn_channels = 4
        cls.heads = 1
        cls.concat = True
        cls.bias = True
        cls.self_attention = True

        return cls

    @classmethod
    def plancksis_gnn_default(cls):
        cls = cls()
        cls.name = "TrainableEpidemics"
        cls.gnn_name = "DynamicsGATConv"
        cls.num_states = 2
        cls.window_size = 1
        cls.window_step = 1
        cls.with_non_edge = False

        cls.loss = "weighted_cross_entropy"
        cls.optimizer = OptimizerConfig.radam_default()

        cls.in_activation = "relu"
        cls.gnn_activation = "relu"
        cls.out_activation = "relu"

        cls.edge_activation = "relu"
        cls.edge_gnn_activation = "relu"

        cls.in_channels = [32]
        cls.gnn_channels = 32
        cls.out_channels = [32]
        cls.edge_channels = [4]
        cls.edge_gnn_channels = 4
        cls.heads = 1
        cls.concat = True
        cls.bias = True
        cls.self_attention = True

        return cls

    @classmethod
    def sissis_gnn_default(cls):
        cls = cls()
        cls.name = "TrainableEpidemics"
        cls.gnn_name = "DynamicsGATConv"
        cls.num_states = 4
        cls.window_size = 1
        cls.window_step = 1
        cls.with_non_edge = False

        cls.loss = "weighted_cross_entropy"
        cls.optimizer = OptimizerConfig.radam_default()

        cls.in_activation = "relu"
        cls.gnn_activation = "relu"
        cls.out_activation = "relu"

        cls.edge_activation = "relu"
        cls.edge_gnn_activation = "relu"

        cls.in_channels = [32, 32]
        cls.gnn_channels = 32
        cls.out_channels = [32, 32]
        cls.edge_channels = [4]
        cls.edge_gnn_channels = 4
        cls.heads = 2
        cls.concat = True
        cls.bias = True
        cls.self_attention = True

        return cls

    @classmethod
    def hidden_sissis_gnn_default(cls):
        cls = cls()
        cls.name = "TrainableEpidemics"
        cls.gnn_name = "DynamicsGATConv"
        cls.num_states = 2
        cls.window_size = 1
        cls.window_step = 1
        cls.with_non_edge = True

        cls.loss = "weighted_cross_entropy"
        cls.optimizer = OptimizerConfig.radam_default()

        cls.in_activation = "relu"
        cls.gnn_activation = "relu"
        cls.out_activation = "relu"

        cls.edge_activation = "relu"
        cls.edge_gnn_activation = "relu"

        cls.in_channels = [32, 32]
        cls.gnn_channels = 32
        cls.out_channels = [32, 32]
        cls.edge_channels = [4]
        cls.edge_gnn_channels = 4
        cls.heads = 2
        cls.concat = True
        cls.bias = True
        cls.self_attention = True
        cls.with_non_edge = False

        return cls

    @classmethod
    def partially_hidden_sissis_gnn_default(cls):
        cls = cls()
        cls.name = "TrainableEpidemics"
        cls.gnn_name = "DynamicsGATConv"
        cls.num_states = 4
        cls.window_size = 1
        cls.window_step = 1
        cls.with_non_edge = True

        cls.loss = "weighted_cross_entropy"
        cls.optimizer = OptimizerConfig.radam_default()

        cls.in_activation = "relu"
        cls.gnn_activation = "relu"
        cls.out_activation = "relu"

        cls.edge_activation = "relu"
        cls.edge_gnn_activation = "relu"

        cls.in_channels = [32, 32]
        cls.gnn_channels = 32
        cls.out_channels = [32, 32]
        cls.edge_channels = [4]
        cls.edge_gnn_channels = 4
        cls.heads = 2
        cls.concat = True
        cls.bias = True
        cls.self_attention = True
        cls.with_non_edge = False

        return cls

    @classmethod
    def metasis_gnn_default(cls):
        cls = cls()
        cls.name = "TrainableMetaPop"
        cls.gnn_name = "DynamicsGATConv"
        cls.num_states = 2
        cls.window_size = 1
        cls.window_step = 1

        cls.loss = "weighted_mse"
        cls.optimizer = OptimizerConfig.radam_default()

        cls.weighted = True

        cls.in_activation = "relu"
        cls.gnn_activation = "relu"
        cls.edge_activation = "relu"
        cls.edge_gnn_activation = "relu"
        cls.out_activation = "relu"

        cls.in_channels = [64, 64, 64]
        cls.gnn_channels = 64
        cls.edge_channels = [32, 32]
        cls.edge_gnn_channels = 32
        cls.out_channels = [64, 64, 64]
        cls.heads = 4
        cls.concat = True
        cls.bias = True
        cls.self_attention = True
        cls.using_log = False

        return cls

    @classmethod
    def metasir_gnn_default(cls):
        cls = cls()
        cls.name = "TrainableMetaPop"
        cls.gnn_name = "DynamicsGATConv"
        cls.num_states = 3
        cls.window_size = 1
        cls.window_step = 1

        cls.loss = "weighted_mse"
        cls.optimizer = OptimizerConfig.radam_default()

        cls.weighted = True

        cls.in_activation = "relu"
        cls.gnn_activation = "relu"
        cls.edge_activation = "relu"
        cls.edge_gnn_activation = "relu"
        cls.out_activation = "relu"

        cls.in_channels = [32, 32, 32, 32]
        cls.gnn_channels = 64
        cls.edge_channels = [64, 64]
        cls.edge_gnn_channels = 64
        cls.out_channels = [32, 32, 32, 32]
        cls.heads = 8
        cls.concat = True
        cls.bias = True
        cls.self_attention = True

        cls.using_log = False

        return cls
