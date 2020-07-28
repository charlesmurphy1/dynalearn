import torch
import torch.nn as nn
import torch_geometric

from torch.nn import Parameter
from .gat import DynamicsGATConv
from .gnn import GraphNeuralNetwork
from dynalearn.config import Config
from dynalearn.nn.activation import get as get_activation
from torch.nn.init import kaiming_normal_
from torch_geometric.utils import to_scipy_sparse_matrix


class WeightedEpidemicsGNN(GraphNeuralNetwork):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        GraphNeuralNetwork.__init__(self, config=config, **kwargs)
        self.num_states = config.num_states
        self.window_size = config.window_size
        self.in_channels = config.in_channels
        self.att_channels = config.att_channels
        self.out_channels = config.out_channels
        self.edge_channels = (
            config.edge_channels if "edge_channels" in config.__dict__ else None
        )
        self.edge_att_channels = (
            config.edge_att_channels if "edge_att_channels" in config.__dict__ else None
        )
        self.heads = config.heads
        self.concat = config.concat
        self.bias = config.bias
        self.attn_bias = config.attn_bias
        self.self_attention = config.self_attention
        self.in_activation = get_activation(config.in_activation)
        self.att_activation = get_activation(config.att_activation)
        self.out_activation = get_activation(config.out_activation)

        self.edge_activation = (
            get_activation(config.edge_activation)
            if "edge_activation" in config.__dict__
            else None
        )
        self.edge_att_activation = (
            get_activation(config.edge_att_activation)
            if "edge_att_activation" in config.__dict__
            else None
        )

        in_layer_channels = [self.window_size, *self.in_channels]
        self.in_layers = self._build_layer(
            in_layer_channels, self.in_activation, bias=self.bias
        )
        edge_layer_channels = [1, *self.edge_channels]
        self.edge_layers = self._build_layer(
            edge_layer_channels, self.edge_activation, bias=self.bias
        )

        self.att_layer = DynamicsGATConv(
            self.in_channels[-1],
            self.att_channels,
            heads=self.heads,
            concat=self.concat,
            bias=self.bias,
            attn_bias=self.attn_bias,
            edge_in_channels=self.edge_channels[-1],
            edge_out_channels=self.edge_att_channels,
            self_attention=self.self_attention,
        )

        if self.concat:
            out_layer_channels = [self.heads * self.att_channels, *self.out_channels]
        else:
            out_layer_channels = [self.att_channels, *self.out_channels]
        self.out_layers = self._build_layer(
            out_layer_channels, self.out_activation, bias=self.bias
        )
        self.last_layer = nn.Linear(
            self.out_channels[-1], self.num_states, bias=self.bias
        )
        self.reset_parameter()
        self.optimizer = self.optimizer(self.parameters())
        if torch.cuda.is_available():
            self = self.cuda()

    def forward(self, x, edge_index, edge_attr=None):
        x = x.T
        x = self.in_layers(x)
        if isinstance(edge_attr, torch.Tensor):
            edge_attr = self.edge_layers(edge_attr)
            x, edge_attr = self.att_layer(x, edge_index, edge_attr=edge_attr)
        else:
            x = self.att_layer(x, edge_index)
        x = self.out_layers(x)
        x = self.last_layer(x)
        return torch.softmax(x, dim=-1)

    def reset_parameter(self, initialize_inplace=None):
        if initialize_inplace is None:
            initialize_inplace = kaiming_normal_

        for layer in self.in_layers:
            if isinstance(layer, nn.Linear):
                initialize_inplace(layer.weight)
                if self.bias:
                    layer.bias.data.fill_(0)

        for layer in self.out_layers:
            if isinstance(layer, nn.Linear):
                initialize_inplace(layer.weight)
                if self.bias:
                    layer.bias.data.fill_(0)

        self.att_layer.reset_parameter()

    def _build_layer(self, channels, activation, bias=True):
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Linear(channels[i], channels[i + 1], bias=bias))
            layers.append(activation)

        return nn.Sequential(*layers)
