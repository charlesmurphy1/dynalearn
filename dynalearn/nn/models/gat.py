import torch
import torch.nn as nn

from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import degree


class GraphAttention(MessagePassing):
    def __init__(
        self, in_channels, out_channels, heads=1, concat=True, bias=True, **kwargs
    ):
        super(GraphAttention, self).__init__(aggr="add", **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.bias = bias

        self.linear = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.att_weight = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        if bias:
            self.att_bias = Parameter(torch.Tensor(1, heads))
        else:
            self.register_parameter("att_bias", None)

        self.reset_parameter()

    def reset_parameter(self):
        glorot(self.linear.weight)
        glorot(self.att_weight)

        zeros(self.linear.bias)
        zeros(self.att_bias)

    def forward(self, x, edge_index):
        if torch.is_tensor(x):
            x = self.linear(x)
        else:
            x = (
                None if x[0] is None else self.linear(x[0]),
                None if x[1] is None else self.linear(x[1]),
            )
        agg_features = self.propagate(edge_index, x=x)
        return x + agg_features

    def message(self, x_i, x_j):
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att_weight[:, :, self.out_channels :]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att_weight).sum(dim=-1)

        if self.bias:
            alpha += self.att_bias
        alpha = torch.sigmoid(alpha)
        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        return aggr_out

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )
