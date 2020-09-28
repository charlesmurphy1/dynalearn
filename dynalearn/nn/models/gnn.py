import torch
import torch.nn as nn

from torch.nn import Parameter, Sequential, Linear, Identity
from .gat import DynamicsGATConv
from .model import Model
from .utils import (
    get_in_layers,
    get_out_layers,
    get_edge_layers,
    reset_layer,
    MultiplexLayer,
    ParallelLayer,
)
from torch.nn.init import kaiming_normal_
from dynalearn.nn.activation import get as get_activation
from dynalearn.nn.models.getter import get as get_gnn_layer
from dynalearn.nn.loss import weighted_cross_entropy
from dynalearn.nn.transformers import BatchNormalizer
from dynalearn.config import Config


class GraphNeuralNetwork(Model):
    def __init__(
        self,
        in_size,
        out_size,
        window_size=1,
        nodeattr_size=0,
        edgeattr_size=0,
        layers=None,
        out_act="identity",
        normalize=False,
        config=None,
        **kwargs
    ):
        Model.__init__(self, config=config, **kwargs)

        self.in_size = in_size
        self.out_size = out_size
        self.window_size = window_size
        self.nodeattr_size = nodeattr_size
        self.edgeattr_size = edgeattr_size

        self.first_layer = Sequential(
            Linear(
                self.window_size * self.in_size + self.nodeattr_size,
                self.config.in_channels[0],
                bias=self.config.bias,
            ),
            get_activation(self.config.in_activation),
        )

        self.in_layers = get_in_layers(self.config)
        self.gnn_layer = get_gnn_layer(
            self.config.in_channels[-1], self.config.gnn_channels, self.config
        )
        self.gnn_activation = get_activation(self.config.gnn_activation)
        self.out_layers = get_out_layers(self.config)
        self.last_layer = Sequential(
            Linear(self.config.out_channels[-1], self.out_size, bias=self.config.bias,),
            get_activation(out_act),
        )
        self.edge_layers = Sequential(Identity())
        if normalize:
            input_size = in_size
            target_size = out_size
        else:
            input_size = 0
            target_size = 0
        self.transformers = BatchNormalizer(
            input_size=input_size,
            target_size=target_size,
            edge_size=edgeattr_size,
            node_size=nodeattr_size,
            layers=layers,
        )

        self.reset_parameters()
        self.optimizer = self.get_optimizer(self.parameters())
        if torch.cuda.is_available():
            self = self.cuda()

    def forward(self, x, network_attr):
        edge_index, edge_attr, node_attr = network_attr
        edge_attr = self.edge_layers(edge_attr)
        x = x.view(-1, self.in_size * self.window_size)
        x = self.merge_nodeattr(x, node_attr)
        x = self.first_layer(x)
        x = self.in_layers(x)
        x = self.gnn_layer(x, edge_index, edge_attr=edge_attr)
        if isinstance(x, tuple):
            x = x[0]
        x = self.gnn_activation(x)
        x = self.out_layers(x)
        x = self.last_layer(x)
        return x

    def reset_parameters(self, initialize_inplace=None):
        if initialize_inplace is None:
            initialize_inplace = kaiming_normal_
        reset_layer(self.edge_layers, initialize_inplace=initialize_inplace)
        reset_layer(self.first_layer, initialize_inplace=initialize_inplace)
        reset_layer(self.in_layers, initialize_inplace=initialize_inplace)
        reset_layer(self.out_layers, initialize_inplace=initialize_inplace)
        reset_layer(self.last_layer, initialize_inplace=initialize_inplace)
        self.gnn_layer.reset_parameters()

    def merge_nodeattr(self, x, node_attr):
        if node_attr is None:
            return x
        assert x.shape[0] == node_attr.shape[0]
        n = x.shape[0]
        return torch.cat([x, node_attr.view(n, -1)], dim=-1)


class WeightedGraphNeuralNetwork(GraphNeuralNetwork):
    def __init__(
        self,
        in_size,
        out_size,
        window_size=1,
        nodeattr_size=0,
        edgeattr_size=0,
        out_act="identity",
        normalize=False,
        config=None,
        **kwargs
    ):
        GraphNeuralNetwork.__init__(
            self,
            in_size,
            out_size,
            window_size=window_size,
            nodeattr_size=nodeattr_size,
            out_act=out_act,
            normalize=normalize,
            config=config,
            **kwargs
        )

        # Getting layers
        self.edge_layers = get_edge_layers(edgeattr_size, self.config)
        self.gnn_layer = DynamicsGATConv(
            self.config.in_channels[-1],
            self.config.gnn_channels,
            heads=self.config.heads,
            concat=self.config.concat,
            bias=self.config.bias,
            edge_in_channels=self.config.edge_channels[-1],
            edge_out_channels=self.config.edge_gnn_channels,
            self_attention=self.config.self_attention,
        )

        # Finishing initialization
        self.reset_parameters()
        self.optimizer = self.get_optimizer(self.parameters())
        if torch.cuda.is_available():
            self = self.cuda()


class MultiplexGraphNeuralNetwork(GraphNeuralNetwork):
    def __init__(
        self,
        in_size,
        out_size,
        window_size=1,
        nodeattr_size=0,
        out_act="identity",
        normalize=False,
        config=None,
        **kwargs
    ):
        GraphNeuralNetwork.__init__(
            self,
            in_size,
            out_size,
            window_size=window_size,
            nodeattr_size=nodeattr_size,
            out_act=out_act,
            normalize=normalize,
            layers=config.network_layers,
            config=config,
            **kwargs
        )
        template = lambda: DynamicsGATConv(
            self.config.in_channels[-1],
            self.config.gnn_channels,
            heads=self.config.heads,
            concat=self.config.concat,
            bias=self.config.bias,
            self_attention=self.config.self_attention,
        )
        self.gnn_layer = MultiplexLayer(
            template, self.config.network_layers, merge="mean"
        )

        # Finishing initialization
        self.reset_parameters()
        self.optimizer = self.get_optimizer(self.parameters())
        if torch.cuda.is_available():
            self = self.cuda()

    def merge_nodeattr(self, x, node_attr):
        if self.nodeattr_size > 0:
            for k, v in node_attr.items():
                if v is not None:
                    x = torch.cat([x, v.view(-1, self.nodeattr_size)], dim=-1)
        return x


class WeightedMultiplexGraphNeuralNetwork(MultiplexGraphNeuralNetwork):
    def __init__(
        self,
        in_size,
        out_size,
        window_size=1,
        nodeattr_size=0,
        edgeattr_size=0,
        out_act="identity",
        normalize=False,
        config=None,
        **kwargs
    ):
        MultiplexGraphNeuralNetwork.__init__(
            self,
            in_size,
            out_size,
            window_size=window_size,
            nodeattr_size=nodeattr_size,
            out_act=out_act,
            normalize=normalize,
            config=config,
            **kwargs
        )
        template = lambda: DynamicsGATConv(
            self.config.in_channels[-1],
            self.config.gnn_channels,
            heads=self.config.heads,
            concat=self.config.concat,
            bias=self.config.bias,
            self_attention=self.config.self_attention,
        )
        self.gnn_layer = MultiplexLayer(
            template, self.config.network_layers, merge="mean"
        )
        self.edge_layers = ParallelLayer(
            lambda: get_edge_layers(edgeattr_size, self.config),
            keys=self.config.network_layers,
        )

        # Finishing initialization
        self.reset_parameters()
        self.optimizer = self.get_optimizer(self.parameters())
        if torch.cuda.is_available():
            self = self.cuda()
