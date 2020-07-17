import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import degree
from torch_geometric.typing import OptPairTensor, Adj, Size, NoneType, OptTensor
from typing import Union, Tuple, Optional


class MultiHeadLinear(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, bias=False):
        nn.Module.__init__(self)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        self.weight = Parameter(torch.Tensor(out_channels, heads, in_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels, heads))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        x = (x * self.weight).sum(dim=-1)
        if self.bias is not None:
            return x + self.bias
        else:
            return x

    def reset_parameter(self):
        glorot(self.weight)
        zeros(self.bias)

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )


class GraphAttention(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        bias: bool = True,
        attn_bias: bool = True,
        edge_in_channels: int = 0,
        edge_out_channels: int = 0,
        self_attention: bool = True,
        **kwargs,
    ):
        super(GraphAttention, self).__init__(aggr="add", **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.bias = bias
        self.attn_bias = attn_bias
        self.edge_in_channels = edge_in_channels
        self.edge_out_channels = edge_out_channels
        self.self_attention = self_attention
        self._alpha = None

        if isinstance(in_channels, int):
            self.linear_source = nn.Linear(in_channels, heads * out_channels, bias=bias)
            self.linear_target = self.linear_source
        else:
            self.linear_source = nn.Linear(
                in_channels[0], heads * out_channels, bias=bias
            )
            self.linear_target = nn.Linear(
                in_channels[1], heads * out_channels, bias=bias
            )

        if edge_in_channels > 0 and edge_out_channels > 0:
            self.linear_edge = nn.Linear(
                edge_in_channels, heads * edge_out_channels, bias=bias
            )
            self.edge_combine = nn.Linear(
                (edge_out_channels + 1) * heads,
                edge_out_channels * heads,
                bias=attn_bias,
            )
            self.attn_edge = MultiHeadLinear(
                edge_out_channels, 1, heads=heads, bias=attn_bias
            )

        else:
            self.register_parameter("linear_edge", None)
            self.register_parameter("edge_combine", None)
            self.register_parameter("attn_edge", None)

        self.attn_source = MultiHeadLinear(out_channels, 1, heads=heads, bias=attn_bias)
        self.attn_target = MultiHeadLinear(out_channels, 1, heads=heads, bias=attn_bias)

        if self_attention:
            self.self_attn_source = MultiHeadLinear(
                out_channels, 1, heads=heads, bias=attn_bias
            )
            self.self_attn_target = MultiHeadLinear(
                out_channels, 1, heads=heads, bias=attn_bias
            )
        else:
            self.register_parameter("self_attn_source", None)
            self.register_parameter("self_attn_target", None)

        self.reset_parameter()

    def reset_parameter(self):
        glorot(self.linear_source.weight)
        zeros(self.linear_source.bias)
        glorot(self.linear_target.weight)
        zeros(self.linear_target.bias)
        if self.linear_edge is not None:
            glorot(self.linear_edge.weight)
            zeros(self.linear_edge.bias)
            glorot(self.edge_combine.weight)
            zeros(self.edge_combine.bias)
            self.attn_edge.reset_parameter()

        self.attn_source.reset_parameter()
        self.attn_target.reset_parameter()

        if self.self_attn_source is not None:
            self.self_attn_source.reset_parameter()
        if self.self_attn_target is not None:
            self.self_attn_target.reset_parameter()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: Union[Tensor, NoneType] = None,
        return_attention_weights: bool = False,
    ):
        H, C = self.heads, self.out_channels
        x_s: OptTensor = None
        x_t: OptTensor = None
        alpha_s: OptTensor = None
        alpha_t: OptTensor = None
        alpha_e: OptTensor = None
        self_alpha_s: OptTensor = None
        self_alpha_t: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in `GraphAttention`."
            x_s = x_t = self.linear_source(x).view(-1, H, C)
            alpha_s = alpha_t = self.attn_source(x_s)
            if self.self_attention:
                self_alpha_s = self_alpha_t = self.self_attn_source(x_s)
            x_s = x_s.view(-1, H * C)
            x_t = x_s.view(-1, H * C)
        else:
            x_s, x_t = x[0], x[1]
            assert x[0].dim() == 2, "Static graphs not supported in `GATConv`."
            x_s = self.linear_source(x_s).view(-1, H, C)
            alpha_s = self.attn_source(x_s)
            if self.self_attention:
                self_alpha_s = self.self_attn_source(x_s)
            x_s = x_s.view(-1, H * C)
            if x_t is not None:
                x_t = self.linear_target(x_t).view(-1, H, C)
                alpha_t = self.attn_target(x_t)
                if self.self_attention:
                    self_alpha_t = self.self_attn_target(x_t)
                x_t = x_s.view(-1, H * C)

        assert x_s is not None
        assert alpha_s is not None

        if self.linear_edge is not None:
            assert edge_attr is not None
            edge_attr = self.linear_edge(edge_attr).view(-1, H, self.edge_out_channels)
            alpha_e = self.attn_edge(edge_attr)

        # propagation
        out = self.propagate(
            edge_index, x=(x_s, x_t), alpha=(alpha_s, alpha_t), edge_attn=alpha_e,
        ).view(-1, H, C)

        # adding self attention
        if self.self_attention:
            out += torch.sigmoid(self_alpha_t + self_alpha_s).unsqueeze(-1) * x_t.view(
                -1, H, C
            )

        alpha = self._alpha
        self._alpha = None

        # combining edge attributes with attention coefficients
        if self.linear_edge is not None:
            assert edge_attr is not None
            edge_attr = torch.cat([edge_attr, alpha.unsqueeze(-1)], axis=-1)
            edge_attr = edge_attr.view(-1, H * (self.edge_out_channels + 1))
            out_edge = self.edge_combine(edge_attr).view(-1, H, self.edge_out_channels)
        else:
            out_edge = None

        # combining attention heads
        if self.concat:
            out = out.view(-1, H * C)
            if out_edge is not None:
                out_edge = out_edge.view(-1, H * self.edge_out_channels)
        else:
            out = out.mean(dim=1)
            if out_edge is not None:
                out_edge = out_edge.mean(dim=1)

        # return
        if return_attention_weights:
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            if out_edge is not None:
                return out, out_edge
            else:
                return out

    def message(
        self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor, edge_attn: OptTensor,
    ) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = alpha if edge_attn is None else alpha + edge_attn
        alpha = torch.sigmoid(alpha)
        self._alpha = alpha
        x_j = x_j.view(-1, self.heads, self.out_channels)
        return (x_j * alpha.unsqueeze(-1)).view(-1, self.heads * self.out_channels)

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )
