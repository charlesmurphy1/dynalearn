import math
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter


class CompleteModel(nn.Module):
    def __init__(self, graph, n_hidden):
        super(CompleteModel, self).__init__()
        if type(n_hidden) == int: n_hidden = [n_hidden]

        self.n_nodes = graph.number_of_nodes()
        self.n_hidden = n_hidden
        self.n_states = 1

        # Functions
        relu = nn.ReLU()
    
        # Decoder: Generative network weights
        layers = [nn.Linear(self.n_nodes, self.n_hidden[0]),
                  nn.BatchNorm1d(self.n_hidden[0]),
                  relu]
        for i in range(1, len(self.n_hidden)):

            layers.append(nn.Linear(self.n_hidden[i - 1], self.n_hidden[i]))
            layers.append(nn.BatchNorm1d(self.n_hidden[i]))
            layers.append(relu)

        last_layer = nn.Linear(self.n_hidden[-1], self.n_nodes)
        self.model = nn.Sequential(*layers, last_layer)

    def forward(self, input, adj):
        outputs = self.model(input)
        return torch.sigmoid(outputs)


class CompleteMultiStateModel(nn.Module):
    def __init__(self, graph, n_hidden, n_states):
        super(CompleteMultiStateModel, self).__init__()
        if type(n_hidden) == int: n_hidden = [n_hidden]

        self.n_nodes = graph.number_of_nodes()
        self.n_hidden = n_hidden
        self.n_states = n_states

        # Functions
        relu = nn.ReLU()
    
        # Decoder: Generative network weights
        layers = [nn.Linear(self.n_nodes, self.n_hidden[0])]
        for i in range(1, len(self.n_hidden)):

            layers.append(nn.Linear(self.n_hidden[i - 1], self.n_hidden[i]))
            layers.append(nn.BatchNorm1d(self.n_hidden[i]))
            layers.append(relu)

        last_layer = nn.Linear(self.n_hidden[-1], self.n_states * self.n_nodes)
        self.model = nn.Sequential(*layers, last_layer)

    def forward(self, input, adj):
        batch_size = input.size(0)
        outputs = self.model(input).contiguous().view(batch_size, 
                                                     self.n_states,
                                                     self.n_nodes)
        return F.softmax(outputs, dim=1)


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.Tensor(1, in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def get_mask(self, adj):
        n_nodes = adj.shape[0]
        np.fill_diagonal(adj, 1)

        deg = np.zeros([n_nodes, n_nodes])
        np.fill_diagonal(deg, (np.sum(adj, 0))**(-0.5))

        adj = torch.tensor(adj)
        deg = torch.tensor(deg)

        mask = torch.matmul(torch.matmul(deg, adj), deg)
        return mask.view(1, n_nodes, n_nodes)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, adj):
        mask = self.get_mask(adj)
        batch_size = input.size(0)
        outputs = torch.matmul(torch.matmul(mask, input), self.weight)
        if self.bias is not None:
            return outputs + self.bias
        else:
            return outputs

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class GCNModel(nn.Module):
    def __init__(self, n_hidden):
        super(GCNModel, self).__init__()
        if type(n_hidden) == int: n_hidden = [n_hidden]

        self.n_hidden = n_hidden
        self.n_states = 1

        # Functions
        relu = nn.ReLU()
    
        layers = [nn.GraphConv(1, n_hidden[0], bias=False),
                  relu]
        for i in range(1, len(self.n_hidden)):

            layers.append(nn.GraphConv(self.n_hidden[i - 1],
                                          self.n_hidden[i],
                                          bias=False))
            layers.append(relu)

        last_layer = nn.GraphConv(self.n_hidden[-1],
                                     self.n_states,
                                     bias=False)
        self.model = nn.Sequential(*layers, last_layer)

    def forward(self, inputs, adj):
        outputs = self.model(inputs, adj)
        return torch.sigmoid(outputs)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True,
                 min_value=1e-15, with_self_attn=False):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.min_value = min_value
        self.with_self_attn = with_self_attn

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))

        nn.init.xavier_uniform_(self.W.data, gain=math.sqrt(2))
        nn.init.xavier_uniform_(self.a.data, gain=math.sqrt(2))



        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.elu = nn.ELU()

    def forward(self, inputs, adj):
        h = torch.matmul(inputs, self.W)
        batch_size = h.size(0)
        N = h.size(1)

        h1 = h.repeat(1, 1, N).view(batch_size, N * N, self.out_features)
        h2 = h.repeat(1, N, 1)
        a_input = torch.cat([h1, h2], dim=2).view(batch_size, N, N, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        if self.with_self_attn:
            adj += torch.eye(N)

        zero_vec = self.min_value*torch.ones_like(e)

        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)

        outputs = torch.matmul(attention, h)

        if self.concat:
            return self.elu(outputs)
        else:
            return outputs

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MultiHeadGAT(nn.Module):
    def __init__(self, in_features, out_features, n_heads, dropout=0.6, 
                 alpha=0.2, min_value=1e-15, with_self_attn=False):
        super(MultiHeadGAT, self).__init__()
        self.dropout = dropout
        self.attentions = []
        for i in range(n_heads):
            att = GraphAttentionLayer(in_features,
                                      out_features,
                                      dropout=dropout,
                                      alpha=alpha,
                                      concat=True,
                                      min_value=min_value,
                                      with_self_attn=with_self_attn)
            self.attentions.append(att)
            self.add_module('attention_{}'.format(i), att)

        # self.out_att = GraphAttentionLayer(out_features * n_heads,
        #                                    out_features,
        #                                    dropout=dropout,
        #                                    alpha=alpha,
        #                                    concat=False,
        #                                    min_value=min_value,
        #                                    with_self_attn=with_self_attn)
        self.out_layer = nn.Parameter(torch.zeros(size=(out_features * n_heads,
                                                        out_features)))
        nn.init.xavier_uniform_(self.out_layer.data, gain=math.sqrt(2))


    def forward(self, inputs, adj):
        x = F.dropout(inputs, self.dropout, training=self.training)
        h = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        h = F.dropout(h, self.dropout, training=self.training)
        # outputs = F.elu(self.out_att(h, adj))
        outputs = F.elu(torch.matmul(h, self.out_layer))
        return outputs


class GATModel(nn.Module):
    def __init__(self, n_hidden, n_heads, dropout, alpha,
                 min_value=1e-15, with_self_attn=False):
        super(GATModel, self).__init__()
        if type(n_hidden) == int: n_hidden = [n_hidden]
        if type(n_heads) == int: n_heads = [n_heads] * len(n_hidden)

        self.n_hidden = [1] + n_hidden
        self.n_heads = n_heads

        self.layers = []
        for i in range(0, len(self.n_hidden) - 1):
            att = MultiHeadGAT(self.n_hidden[i],
                               self.n_hidden[i + 1],
                               self.n_heads[i],
                               dropout,
                               alpha,
                               min_value,
                               with_self_attn)
            self.layers.append(att)
            self.add_module('multhead_attention_{}'.format(i), att)
        self.out_layer = nn.Parameter(torch.zeros(size=(self.n_hidden[-1], 1)))
        nn.init.xavier_uniform_(self.out_layer.data, gain=math.sqrt(2))

    def forward(self, inputs, adj):
        for layer in self.layers:
            inputs = layer(inputs, adj)
        outputs = torch.matmul(inputs, self.out_layer)
        return torch.sigmoid(outputs)