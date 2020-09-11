from .config import Config
from .optimizers import OptimizerConfig
from dynalearn.nn.optimizers import *


class TrainableConfig(Config):
    @classmethod
    def test(cls):
        cls = cls()
        cls.name = "TrainableEpidemics"
        cls.num_states = 2
        cls.window_size = 1
        cls.window_step = 1

        cls.optimizer = OptimizerConfig.radam()

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
    def sis(cls):
        cls = cls()
        cls.name = "TrainableStochasticEpidemics"
        cls.gnn_name = "DynamicsGATConv"
        cls.num_states = 2
        cls.window_size = 1
        cls.window_step = 1

        cls.optimizer = OptimizerConfig.radam()

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
    def plancksis(cls):
        cls = cls()
        cls.name = "TrainableEpidemics"
        cls.gnn_name = "DynamicsGATConv"
        cls.num_states = 2
        cls.window_size = 1
        cls.window_step = 1

        cls.optimizer = OptimizerConfig.radam()

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
    def sissis(cls):
        cls = cls()
        cls.name = "TrainableEpidemics"
        cls.gnn_name = "DynamicsGATConv"
        cls.num_states = 4
        cls.window_size = 1
        cls.window_step = 1

        cls.optimizer = OptimizerConfig.radam()

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
    def hidden_sissis(cls):
        cls = cls()
        cls.name = "TrainableEpidemics"
        cls.gnn_name = "DynamicsGATConv"
        cls.num_states = 2
        cls.window_size = 1
        cls.window_step = 1
        cls.with_non_edge = True

        cls.optimizer = OptimizerConfig.radam()

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
    def partially_hidden_sissis(cls):
        cls = cls()
        cls.name = "TrainableEpidemics"
        cls.gnn_name = "DynamicsGATConv"
        cls.num_states = 4
        cls.window_size = 1
        cls.window_step = 1
        cls.with_non_edge = True

        cls.optimizer = OptimizerConfig.radam()

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
    def rdsis(cls):
        cls = cls()
        cls.name = "TrainableReactionDiffusion"
        cls.gnn_name = "DynamicsGATConv"
        cls.num_states = 2
        cls.window_size = 1
        cls.window_step = 1

        cls.alpha = np.array([0.5, 0.5])
        cls.optimizer = OptimizerConfig.radam()

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

        return cls

    @classmethod
    def rdsir(cls):
        cls = cls()
        cls.name = "TrainableReactionDiffusion"
        cls.gnn_name = "DynamicsGATConv"
        cls.num_states = 3
        cls.window_size = 1
        cls.window_step = 1

        cls.alpha = np.array([0.5, 0.5])
        cls.optimizer = OptimizerConfig.radam()

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

        return cls

    @classmethod
    def dsis(cls):
        cls = cls()
        cls.name = "TrainableDeterministicEpidemics"
        cls.gnn_name = "DynamicsGATConv"
        cls.num_states = 2
        cls.window_size = 1
        cls.window_step = 1

        cls.optimizer = OptimizerConfig.radam()

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

        return cls

    @classmethod
    def dsir(cls):
        cls = cls()
        cls.name = "TrainableDeterministicEpidemics"
        cls.gnn_name = "DynamicsGATConv"
        cls.num_states = 3
        cls.window_size = 1
        cls.window_step = 1

        cls.optimizer = OptimizerConfig.radam()

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

        return cls
