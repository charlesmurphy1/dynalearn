import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import Parameter, Linear, Sequential, Module
from torch.nn.init import kaiming_normal_
from torch_geometric.nn.inits import glorot, zeros
from dynalearn.nn.activation import get as get_activation

__rnn_layers__ = {
    "RNN": lambda config: torch.nn.RNN(
        input_size=config.hidden_channels[0],
        hidden_size=config.hidden_channels[-1],
        num_layers=config.num_layers,
        nonlinearity=config.activation,
        bias=config.bias,
        bidirectional=config.bidirectional,
    ),
    "LSTM": lambda config: torch.nn.LSTM(
        input_size=config.hidden_channels[0],
        hidden_size=config.hidden_channels[-1],
        num_layers=config.num_layers,
        bias=config.bias,
        bidirectional=config.bidirectional,
    ),
    "GRU": lambda config: torch.nn.GRU(
        input_size=config.hidden_channels[0],
        hidden_size=config.hidden_channels[-1],
        num_layers=config.num_layers,
        bias=config.bias,
        bidirectional=config.bidirectional,
    ),
}


class ParallelLayer(nn.Module):
    def __init__(self, template, keys, merge="None"):
        nn.Module.__init__(self)
        self.template = template
        self.keys = keys
        self.merge = merge

        for k in self.keys:
            setattr(self, f"layer_{k}", template())

    def forward(self, x, **kwargs):
        y = None
        for k in self.keys:
            yy = getattr(self, f"layer_{k}")(x[k], **kwargs)
            if y is None:
                if isinstance(yy, tuple):
                    y = ({k: yyy} for yyy in yy)
                else:
                    y = {k: yy}
            else:
                if isinstance(yy, tuple):
                    for i, yyy in enumerate(yy):
                        y[i][k] = yyy
                else:
                    y[k] = yy
        if self.merge == "concat":
            return self._merge_concat_(y)
        elif self.merge == "mean":
            return self._merge_mean_(y)
        elif self.merge == "sum":
            return self._merge_sum_(y)
        else:
            if isinstance(y, list):
                return tuple(y)
            else:
                return y

    def reset_parameters(self, initialize_inplace=None):
        if initialize_inplace is None:
            initialize_inplace = kaiming_normal_
        for k in self.keys:
            layer = getattr(self, f"layer_{k}")
            if isinstance(layer, Sequential):
                for l in layer:
                    if type(l) == Linear:
                        initialize_inplace(l.weight)
                        if l.bias is not None:
                            l.bias.data.fill_(0)
            else:
                try:
                    layer.reset_parameters()
                except:
                    pass

    def __repr__(self):
        name = self.template().__class__.__name__
        return "{}({}, heads={})".format(
            self.__class__.__name__,
            name,
            len(self.keys),
        )

    def _merge_mean_(self, y):
        if isinstance(y, list):
            out = (
                torch.cat(
                    [yy[k].view(*yy[k].shape, 1) for k in self.keys],
                    axis=-1,
                )
                for yy in y
            )
            return (torch.mean(yy, axis=-1) for yy in out)
        else:
            out = torch.cat(
                [y[k].view(*y[k].shape, 1) for k in self.keys],
                axis=-1,
            )
            return torch.mean(out, axis=-1)

    def _merge_sum_(self, y):
        if isinstance(y, list):
            out = (
                torch.cat(
                    [yy[k].view(*yy[k].shape, 1) for k in self.keys],
                    axis=-1,
                )
                for yy in y
            )
            return (torch.mean(yy, axis=-1) for yy in out)
        else:
            out = torch.cat(
                [y[k].view(*y[k].shape, 1) for k in self.keys],
                axis=-1,
            )
            return torch.mean(out, axis=-1)

    def _merge_concat_(self, y):
        if isinstance(y, list):
            return (
                torch.cat(
                    [yy[k].view(*yy[k].shape) for k in self.keys],
                    axis=-1,
                )
                for yy in y
            )
        else:
            return torch.cat(
                [y[k].view(*y[k].shape) for k in self.keys],
                axis=-1,
            )


class MultiplexLayer(ParallelLayer):
    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        if edge_attr is None:
            y = {
                k: getattr(self, f"layer_{k}")(x, edge_index[k], **kwargs)
                for k in self.keys
            }
        else:
            y = None
            for k in self.keys:

                yy = getattr(self, f"layer_{k}")(
                    x, edge_index[k], edge_attr[k], **kwargs
                )
                if y is None:
                    if isinstance(yy, tuple):
                        y = [{k: yyy} for yyy in yy]
                    else:
                        y = {k: yy}
                else:
                    if isinstance(yy, tuple):
                        for i, yyy in enumerate(yy):
                            y[i][k] = yyy
                    else:
                        y[k] = yy
        if self.merge == "concat":
            return self._merge_concat_(y)
        elif self.merge == "mean":
            return self._merge_mean_(y)
        elif self.merge == "sum":
            return self._merge_sum_(y)
        else:
            return y


class MultiHeadLinear(nn.Module):
    def __init__(self, num_channels, heads=1, bias=False):
        nn.Module.__init__(self)
        self.num_channels = num_channels
        self.heads = heads

        self.weight = Parameter(torch.Tensor(heads, num_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(heads))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        x = (x * self.weight).sum(dim=-1)
        if self.bias is not None:
            return x + self.bias
        else:
            return x

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def __repr__(self):
        return "{}({}, heads={})".format(
            self.__class__.__name__, self.num_channels, self.heads
        )


class Reshape(nn.Module):
    def __init__(self, shape):
        self.shape = shape
        nn.Module.__init__(self)

    def forward(self, x):
        return x.view(*self.shape)


def build_layers(channel_seq, activation, bias=True):
    layers = []
    activation = get_activation(activation)
    for i in range(len(channel_seq) - 1):
        layers.append(nn.Linear(channel_seq[i], channel_seq[i + 1], bias=bias))
        layers.append(activation)

    return nn.Sequential(*layers)


def get_in_layers(config):
    return build_layers(config.in_channels, config.in_activation, bias=config.bias)


def get_out_layers(config):
    if config.concat:
        out_layer_channels = [config.heads * config.gnn_channels, *config.out_channels]
    else:
        out_layer_channels = [config.gnn_channels, *config.out_channels]
    return build_layers(out_layer_channels, config.out_activation, bias=config.bias)


def get_edge_layers(edge_size, config):
    edge_layer_channels = [edge_size, *config.edge_channels]
    return build_layers(edge_layer_channels, config.edge_activation, bias=config.bias)


def reset_layer(layer, initialize_inplace=None):
    if initialize_inplace is None:
        initialize_inplace = kaiming_normal_
    assert isinstance(layer, Module)

    if isinstance(layer, Sequential):
        for l in layer:
            if type(l) == Linear:
                initialize_inplace(l.weight)
                if l.bias is not None:
                    l.bias.data.fill_(0)
    else:
        layer.reset_parameters()
