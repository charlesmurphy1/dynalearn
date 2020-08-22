import torch
import torch.nn as nn

from torch.nn import Parameter, Sequential, Linear
from .gat import DynamicsGATConv
from .gnn import GraphNeuralNetwork
from .utils import build_layers
from dynalearn.config import Config
from dynalearn.nn.activation import get as get_activation
from dynalearn.nn.models.getter import get as get_gnn_layer
from torch.nn.init import kaiming_normal_


class EpidemicsGNN(GraphNeuralNetwork):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        GraphNeuralNetwork.__init__(self, config=config, **kwargs)
        self.num_states = config.num_states
        self.window_size = config.window_size

        self.with_non_edge = (
            config.with_non_edge if "with_non_edge" in config.__dict__ else False
        )

        in_layer_channels = [self.window_size, *config.in_channels]
        self.in_layers = build_layers(
            in_layer_channels, config.in_activation, bias=config.bias
        )
        # self.gnn_layer = get_layer(config.in_channels, config.gnn_channels, config)
        self.gnn_layer = get_gnn_layer(
            config.in_channels[-1], config.gnn_channels, config
        )

        if self.with_non_edge:
            self.non_edge_layer = Linear(config.in_channels[-1], 1, bias=config.bias)
        else:
            self.register_parameter("non_edge_layer", None)

        if config.concat:
            out_layer_channels = [
                config.heads * config.gnn_channels,
                *config.out_channels,
            ]
        else:
            out_layer_channels = [config.gnn_channels, *config.out_channels]
        self.out_layers = build_layers(
            out_layer_channels, config.out_activation, bias=config.bias
        )
        self.last_layer = Linear(
            config.out_channels[-1], self.num_states, bias=config.bias
        )
        self.reset_parameters()
        self.optimizer = self.get_optimizer(self.parameters())
        if torch.cuda.is_available():
            self = self.cuda()

    def forward(self, x, edge_index):
        x = self.in_layers(x)
        if self.with_non_edge:
            a = self.non_edge_layer(x)
            a = torch.relu(a)
            a = torch.softmax(a, dim=0)
            x = x + torch.sum(a * x, 0)
        x = self.gnn_layer(x, edge_index)
        x = self.gnn_activation(x)
        x = self.out_layers(x)
        x = self.last_layer(x)
        return torch.softmax(x, dim=-1)

    def reset_parameters(self, initialize_inplace=None):
        if initialize_inplace is None:
            initialize_inplace = kaiming_normal_

        self.reset_layer(self.in_layers, initialize_inplace=initialize_inplace)
        self.reset_layer(self.non_edge_layer, initialize_inplace=initialize_inplace)
        self.reset_layer(self.out_layers, initialize_inplace=initialize_inplace)

        self.gnn_layer.reset_parameters()

    def reset_layer(self, layer, initialize_inplace=None):
        if initialize_inplace is None:
            initialize_inplace = kaiming_normal_
        if layer is None:
            return
        assert isinstance(layer, Sequential)
        for l in layer:
            if type(l) == Linear:
                initialize_inplace(l.weight)
                if l.bias is not None:
                    l.bias.data.fill_(0)
