import torch
import torch.nn as nn

from torch.nn import Parameter
from .gat import DynamicsGATConv
from .gnn import GraphNeuralNetwork
from dynalearn.config import Config
from dynalearn.nn.activation import get as get_activation
from torch.nn.init import kaiming_normal_
from torch_geometric.nn import GATConv, SAGEConv, GCNConv, GraphConv


class EpidemicsGNN(GraphNeuralNetwork):
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
        self.heads = config.heads
        self.concat = config.concat
        self.bias = config.bias
        self.attn_bias = config.attn_bias if "attn_bias" in config.__dict__ else False
        self.self_attention = (
            config.self_attention if "self_attention" in config.__dict__ else False
        )
        self.with_non_edge = (
            config.with_non_edge if "with_non_edge" in config.__dict__ else False
        )
        self.in_activation = get_activation(config.in_activation)
        self.att_activation = get_activation(config.att_activation)
        self.out_activation = get_activation(config.out_activation)

        in_layer_channels = [self.window_size, *self.in_channels]
        self.in_layers = self._build_layer(
            in_layer_channels, self.in_activation, bias=self.bias
        )
        in_layer_channels = [self.window_size, *self.in_channels]
        self.in_edge_layers = self._build_layer(
            in_layer_channels, self.in_activation, bias=self.bias
        )
        if "gnn_layer_name" not in config.__dict__:
            config.gnn_layer_name = "DynamicsGAT"
        if config.gnn_layer_name == "GAT":
            self.att_layer = GATConv(
                self.in_channels[-1],
                self.att_channels,
                heads=self.heads,
                concat=self.concat,
                add_self_loops=self.self_attention,
                bias=self.bias,
            )
        elif config.gnn_layer_name == "SAGE":
            self.att_layer = SAGEConv(
                self.in_channels[-1], self.att_channels, bias=self.bias
            )
        elif config.gnn_layer_name == "GCN":
            self.att_layer = GCNConv(
                self.in_channels[-1],
                self.att_channels,
                bias=self.bias,
                add_self_loops=self.self_attention,
            )
        elif config.gnn_layer_name == "GraphConv":
            self.att_layer = GraphConv(
                self.in_channels[-1], self.att_channels, bias=self.bias, aggr="add"
            )
        else:
            self.att_layer = DynamicsGATConv(
                self.in_channels[-1],
                self.att_channels,
                heads=self.heads,
                concat=self.concat,
                bias=self.bias,
                attn_bias=self.attn_bias,
                self_attention=self.self_attention,
            )

        if self.with_non_edge:
            self.non_edge_layer = nn.Linear(self.in_channels[-1], 1, bias=self.bias)
        else:
            self.register_parameter("non_edge_layer", None)

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
        self.reset_parameters()
        self.optimizer = self.optimizer(self.parameters())
        if torch.cuda.is_available():
            self = self.cuda()

    def forward(self, x, edge_index):
        x = x.T
        x = self.in_layers(x)
        if self.with_non_edge:
            a = self.non_edge_layer(x)
            a = torch.relu(a)
            a = torch.softmax(a, dim=0)
            x = x + torch.sum(a * x, 0)
        x = self.att_layer(x, edge_index)
        x = self.out_layers(x)
        x = self.last_layer(x)
        return torch.softmax(x, dim=-1)

    def reset_parameters(self, initialize_inplace=None):
        if initialize_inplace is None:
            initialize_inplace = kaiming_normal_

        for layer in self.in_layers:
            if type(layer) == torch.nn.Linear:
                initialize_inplace(layer.weight)
                if self.bias:
                    layer.bias.data.fill_(0)

        for layer in self.out_layers:
            if type(layer) == torch.nn.Linear:
                initialize_inplace(layer.weight)
                if self.bias:
                    layer.bias.data.fill_(0)

        self.att_layer.reset_parameters()

    def _build_layer(self, channels, activation, bias=True):
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Linear(channels[i], channels[i + 1], bias=bias))
            layers.append(activation)

        return nn.Sequential(*layers)
