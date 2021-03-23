import torch
import torch.nn as nn

from abc import abstractmethod
from torch.nn import Parameter, Sequential, Linear, Identity, RNN, LSTM, GRU
from .gat import DynamicsGATConv
from .model import Model
from .utils import build_layers, reset_layer, __rnn_layers__
from torch.nn.init import kaiming_normal_
from dynalearn.nn.activation import get as get_activation
from dynalearn.nn.models.getter import get as get_gnn_layer
from dynalearn.nn.transformers import BatchNormalizer
from dynalearn.config import Config


class UnivariateModel(Model):
    def __init__(
        self,
        in_size,
        out_size,
        window_size=1,
        nodeattr_size=0,
        out_act="identity",
        normalize=False,
        config=None,
        **kwargs,
    ):
        Model.__init__(self, config=config, **kwargs)

        self.in_size = in_size
        self.out_size = out_size
        self.window_size = window_size
        self.nodeattr_size = nodeattr_size
        self.out_act = out_act
        self.layers = self.build_layers()
        self.out_layer = Sequential(
            Linear(
                self.config.hidden_channels[-1],
                self.out_size,
                bias=self.config.bias,
            ),
            get_activation(self.out_act),
        )
        if normalize:
            input_size = in_size
            target_size = out_size
        else:
            input_size = 0
            target_size = 0
        self.transformers = BatchNormalizer(
            input_size=input_size, target_size=target_size, node_size=nodeattr_size
        )

        self.reset_parameters()
        self.optimizer = self.get_optimizer(self.parameters())
        if torch.cuda.is_available():
            self = self.cuda()

    @abstractmethod
    def build_layers():
        raise NotImplementedError()

    @abstractmethod
    def format_input(self, x, node_attr=None):
        raise NotImplementedError()

    def forward(self, x, network_attr):
        edge_index, edge_attr, node_attr = network_attr
        x = self.format_input(x, node_attr)
        x = self.layers(x)
        return self.out_layer(x)

    def reset_parameters(self, initialize_inplace=None):
        if initialize_inplace is None:
            initialize_inplace = kaiming_normal_
        reset_layer(self.layers, initialize_inplace=initialize_inplace)
        reset_layer(self.out_layer, initialize_inplace=initialize_inplace)


class UnivariateMPL(UnivariateModel):
    def format_input(self, x, node_attr):
        n = x.shape[0]
        x = x.view(n, -1)  # [batch, features]
        if node_attr is not None:
            x = torch.cat([x, node_attr.view(-1, self.nodeattr_size)], dim=1)
        return x

    def build_layers(self):
        layers = []
        layers.append(
            Sequential(
                Linear(
                    self.window_size * self.in_size + self.nodeattr_size,
                    self.config.hidden_channels[0],
                    bias=self.config.bias,
                ),
                get_activation(self.config.activation),
            )
        )
        layers.append(
            build_layers(
                self.config.hidden_channels,
                self.config.activation,
                bias=self.config.bias,
            )
        )
        return Sequential(*layers)


class UnivariateRNN(UnivariateModel):
    def __init__(
        self,
        *args,
        rnn="RNN",
        **kwargs,
    ):

        if isinstance(rnn, str):
            assert (
                rnn in __rnn_layers__
            ), f"`{rnn}` is invalid, valid entries are `{list(__rnn_layers__.keys())}`."
            self.rnn = __rnn_layers__[rnn]
        else:
            assert issubclass(
                rnn.__class__, torch.nn.RNNBase
            ), f"Invalid `rnn` template, expected an `torch.nn.RNNBase` subclass but received `{rnn.__class__}`."
            self.rnn = rnn
        UnivariateModel.__init__(self, *args, **kwargs)

    def format_input(self, x, node_attr):
        # x: [batch, features, timestamps]
        if node_attr is not None:
            node_attr = node_attr.view(-1, self.nodeattr_size, 1).repeat(
                1, 1, self.window_size
            )
            x = torch.cat([x, node_attr], dim=1)
        x = torch.transpose(x, 0, 1)  # [features, batch, timestamps]
        x = torch.transpose(x, 0, 2)  # [timestamps, batch, features]
        return x

    def forward(self, x, network_attr):
        edge_index, edge_attr, node_attr = network_attr
        x = self.format_input(x, node_attr)
        x, h = self.layers(x)
        return self.out_layer(x)[-1]

    def build_layers(self):
        rnn_layers = [
            Sequential(
                Linear(
                    self.in_size + self.nodeattr_size,
                    self.config.hidden_channels[0],
                    bias=self.config.bias,
                ),
                get_activation(self.config.activation),
            )
        ]
        rnn_layers.append(self.rnn(self.config))
        return Sequential(*rnn_layers)
