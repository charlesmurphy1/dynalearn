import numpy as np
import torch
import torch.nn as nn

from torch.nn import Parameter, Linear, Sequential
from torch.nn.init import kaiming_normal_
from .gat import DynamicsGATConv
from .variant import (
    ContinuousGraphNeuralNetwork,
    ContinuousWeightedGraphNeuralNetwork,
    ContinuousMultiplexGraphNeuralNetwork,
    ContinuousWeightedMultiplexGraphNeuralNetwork,
)
from .utils import (
    get_in_layers,
    get_out_layers,
    get_edge_layers,
    MultiplexLayer,
    ParallelLayer,
)
from dynalearn.config import Config
from dynalearn.nn.activation import get as get_activation
from dynalearn.nn.models.getter import get as get_gnn_layer
from dynalearn.utilities import get_edge_attr


class MetaPopGNN(ContinuousGraphNeuralNetwork):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        self.window_size = config.window_size
        self.num_states = config.num_states
        config.input_shape = torch.Size([1, config.num_states, config.window_size])
        config.target_shape = torch.Size([1, config.num_states])
        ContinuousGraphNeuralNetwork.__init__(self, config=config, **kwargs)

        # Getting layers
        self.in_layers = get_in_layers(config, True)
        self.gnn_layer = get_gnn_layer(
            config.in_channels[-1], config.gnn_channels, config
        )
        self.gnn_activation = get_activation(config.gnn_activation)
        self.out_layers = get_out_layers(config)
        self.last_layer = nn.Linear(
            config.out_channels[-1], config.num_states, bias=config.bias
        )

        # Finishing initialization
        MetaPopGNN.reset_parameters(self)

        self.optimizer = self.get_optimizer(self.parameters())
        if torch.cuda.is_available():
            self = self.cuda()

    def forward(self, x, edge_index, edge_attr=None):
        x = x.view(-1, self.window_size * self.num_states)
        x = self.in_layers(x)
        x, _ = self.gnn_layer(x, edge_index)
        x = self.gnn_activation(x)
        x = self.out_layers(x)
        x = self.last_layer(x)
        return x

    def reset_parameters(self, initialize_inplace=None):
        if initialize_inplace is None:
            initialize_inplace = kaiming_normal_

        self.reset_layer(self.in_layers, initialize_inplace=initialize_inplace)
        self.reset_layer(self.out_layers, initialize_inplace=initialize_inplace)
        self.gnn_layer.reset_parameters()
        self.last_layer.reset_parameters()

    def reset_layer(self, layer, initialize_inplace=None):
        if initialize_inplace is None:
            initialize_inplace = kaiming_normal_
        assert isinstance(layer, Sequential)
        for l in layer:
            if type(l) == Linear:
                initialize_inplace(l.weight)
                if l.bias is not None:
                    l.bias.data.fill_(0)


class MetaPopWGNN(MetaPopGNN, ContinuousWeightedGraphNeuralNetwork):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs

        config.edgeattr_shape = torch.Size([1, 1])
        ContinuousWeightedGraphNeuralNetwork.__init__(self, config=config, **kwargs)
        MetaPopGNN.__init__(self, config=config, **kwargs)

        # Getting layers
        self.edge_layers = get_edge_layers(config)
        self.gnn_layer = DynamicsGATConv(
            config.in_channels[-1],
            config.gnn_channels,
            heads=config.heads,
            concat=config.concat,
            bias=config.bias,
            edge_in_channels=config.edge_channels[-1],
            edge_out_channels=config.edge_gnn_channels,
            self_attention=config.self_attention,
        )

        # Finishing initialization
        MetaPopWGNN.reset_parameters(self)
        self.optimizer = self.get_optimizer(self.parameters())
        if torch.cuda.is_available():
            self = self.cuda()

    def forward(self, x, edge_index, edge_attr=None):
        x = x.view(-1, self.window_size * self.num_states)
        x = self.in_layers(x)
        edge_attr = self.edge_layers(edge_attr)
        x, _ = self.gnn_layer(x, edge_index, edge_attr=edge_attr)
        x = self.gnn_activation(x)
        x = self.out_layers(x)
        x = self.last_layer(x)
        return x

    def reset_parameters(self, initialize_inplace=None):
        if initialize_inplace is None:
            initialize_inplace = kaiming_normal_

        self.reset_layer(self.in_layers, initialize_inplace=initialize_inplace)
        self.reset_layer(self.edge_layers, initialize_inplace=initialize_inplace)
        self.reset_layer(self.out_layers, initialize_inplace=initialize_inplace)
        self.gnn_layer.reset_parameters()
        self.last_layer.reset_parameters()


class MetaPopMGNN(MetaPopGNN, ContinuousMultiplexGraphNeuralNetwork):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        MetaPopGNN.__init__(self, config=config, **kwargs)
        ContinuousMultiplexGraphNeuralNetwork.__init__(self, config=config, **kwargs)

        # Getting layers
        template = lambda: DynamicsGATConv(
            self.in_channels[-1],
            self.gnn_channels,
            heads=self.heads,
            concat=self.concat,
            bias=self.bias,
            self_attention=self.self_attention,
        )
        self.gnn_layer = MultiplexLayer(
            template, network_layers=self.network_layers, merge="mean"
        )
        self.merge_layer = Sequential(
            # Linear(
            #     config.gnn_channels * len(config.network_layers),
            #     config.gnn_channels,
            #     bias=config.bias,
            # ),
            get_activation(config.gnn_activation),
        )

        # Finishing initialization
        MetaPopMGNN.reset_parameters(self)
        self.optimizer = self.get_optimizer(self.parameters())
        if torch.cuda.is_available():
            self = self.cuda()

    def forward(self, x, edge_index):
        x = x.view(-1, self.window_size * self.num_states)
        x = self.in_layers(x)
        x = self.merge_layer(self.gnn_layer(x, edge_index))
        x = self.out_layers(x)
        x = self.last_layer(x)
        return x

    def reset_parameters(self, initialize_inplace=None):
        if initialize_inplace is None:
            initialize_inplace = kaiming_normal_

        self.reset_layer(self.in_layers, initialize_inplace=initialize_inplace)
        self.reset_layer(self.merge_layer, initialize_inplace=initialize_inplace)
        self.reset_layer(self.out_layers, initialize_inplace=initialize_inplace)
        self.gnn_layer.reset_parameters()
        self.last_layer.reset_parameters()


class MetaPopWMGNN(MetaPopGNN, ContinuousWeightedMultiplexGraphNeuralNetwork):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        config.edgeattr_shape = torch.Size([1, 1])

        ContinuousWeightedMultiplexGraphNeuralNetwork.__init__(
            self, config=config, **kwargs
        )
        MetaPopGNN.__init__(self, config=config, **kwargs)

        # Getting layers
        self.edge_layers = ParallelLayer(
            lambda: get_edge_layers(config), keys=self.network_layers
        )
        template = lambda: DynamicsGATConv(
            config.in_channels[-1],
            config.gnn_channels,
            heads=config.heads,
            concat=config.concat,
            bias=config.bias,
            edge_in_channels=config.edge_channels[-1],
            edge_out_channels=config.edge_gnn_channels,
            self_attention=config.self_attention,
        )
        self.gnn_layer = MultiplexLayer(
            template, keys=self.network_layers, merge="mean"
        )
        self.merge_layer = Sequential(get_activation(config.gnn_activation),)

        # Finishing initialization
        MetaPopWMGNN.reset_parameters(self)
        self.optimizer = self.get_optimizer(self.parameters())
        if torch.cuda.is_available():
            self = self.cuda()

    def forward(self, x, edge_index, edge_attr):
        x = x.view(-1, self.window_size * self.num_states)
        x = self.in_layers(x)
        edge_attr = self.edge_layers(edge_attr)
        x, _ = self.gnn_layer(x, edge_index, edge_attr=edge_attr)
        x = self.merge_layer(x)
        x = self.out_layers(x)
        x = self.last_layer(x)
        return x

    def reset_parameters(self, initialize_inplace=None):
        if initialize_inplace is None:
            initialize_inplace = kaiming_normal_

        self.reset_layer(self.in_layers, initialize_inplace=initialize_inplace)
        self.reset_layer(self.merge_layer, initialize_inplace=initialize_inplace)
        self.reset_layer(self.out_layers, initialize_inplace=initialize_inplace)
        self.edge_layers.reset_parameters()
        self.gnn_layer.reset_parameters()
        self.last_layer.reset_parameters()
