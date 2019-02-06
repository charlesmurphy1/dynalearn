import math
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter


class NodeLinear(nn.Module):
    def __init__(self, num_nodes, in_features, out_features, bias=True):
        super(NodeLinear, self).__init__()
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.Tensor(num_nodes,
                                             in_features,
                                             out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(num_nodes,
                                               out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        batch_size = input.size(0)
        input = input.view(batch_size, self.num_nodes, 1, self.in_features)
        ans = torch.matmul(input, self.weight).view(batch_size,
                                                    self.num_nodes,
                                                    self.out_features)
        if self.bias is not None:
            return ans + self.bias
        else:
            return ans

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class NodeEncoder(nn.Module):
    def __init__(self, graph, n_hidden, n_embedding, keepprob=1):
        super(NodeEncoder, self).__init__()
        self.num_nodes = graph.number_of_nodes()
        self.n_hidden = n_hidden
        self.n_embedding = n_embedding

        relu = nn.ReLU()
        tanh = nn.Tanh()
        sigmoid = nn.Sigmoid()
        dropout = nn.Dropout(1 - keepprob)

        layers = [NodeLinear(self.num_nodes,
                                  self.num_nodes + 1,
                                  self.n_hidden[0])]
        for i in range(1, len(n_hidden)):
            layers.append(relu)
            layers.append(dropout)
            layers.append(NodeLinear(self.num_nodes,
                                     self.n_hidden[i - 1], 
                                     self.n_hidden[i]))
        self.encoder = nn.Sequential(*layers, tanh, dropout)
        self.mu = NodeLinear(self.num_nodes,
                             self.n_hidden[-1],
                             self.n_embedding)
        self.var = NodeLinear(self.num_nodes,
                              self.n_hidden[-1],
                              self.n_embedding)


    def forward(self, x):
        x = torch.cat([x[0], x[1]], 2)
        h = self.encoder(x)
        return self.mu(h), self.var(h)


class NodeDecoder(nn.Module):
    def __init__(self, graph, n_hidden, n_embedding, keepprob=1):
        super(NodeDecoder, self).__init__()
        self.num_nodes = graph.number_of_nodes()
        self.n_hidden = n_hidden
        self.n_embedding = n_embedding

        relu = nn.ReLU()
        tanh = nn.Tanh()
        sigmoid = nn.Sigmoid()
        dropout = nn.Dropout(1 - keepprob)

        layers = []
        for i in range(len(n_hidden) - 1, 0, -1):
            layers.append(NodeLinear(self.num_nodes, 
                                     self.n_hidden[i],
                                     self.n_hidden[i - 1]))
            layers.append(relu)
            layers.append(dropout)
        self.decoder = nn.Sequential(NodeLinear(self.num_nodes,
                                                self.n_embedding + self.num_nodes,
                                                self.n_hidden[-1]),
                                     tanh,
                                     dropout,
                                     *layers,
                                     NodeLinear(self.num_nodes,
                                                self.n_hidden[0],
                                                1),
                                     sigmoid)


    def forward(self, z):
        z = torch.cat([z[0], z[1]], 2)
        return self.decoder(z)


class SparseNodeLinear(nn.Module):
    def __init__(self, graph, out_features, bias=True):
        super(SparseNodeLinear, self).__init__()

        self.num_nodes = graph.number_of_nodes()
        self.out_features = out_features


        self.weight = Parameter(torch.Tensor(self.num_nodes,
                                             self.num_nodes + 1,
                                             out_features))
        self.edgeMask = Parameter(torch.Tensor(self.num_nodes,
                                               self.num_nodes + 1),
                                  requires_grad=False)

        self.compute_edgeMask(graph)
        if bias:
            self.bias = Parameter(torch.Tensor(self.num_nodes,
                                               out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def compute_edgeMask(self, graph):
        adj = nx.to_numpy_array(graph)
        np.fill_diagonal(adj, 1)
        adj = np.concatenate([np.ones([self.num_nodes, 1]), adj], 1)
        mask = 1 - torch.tensor(adj).byte()
        self.edgeMask.data = mask.view(self.num_nodes, 1, self.num_nodes + 1)


    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        batch_size = input.size(0)

        input = input.view(batch_size, self.num_nodes, 1, self.num_nodes + 1)
        input.masked_fill_(self.edgeMask, 0)
        ans = torch.matmul(input, self.weight).view(batch_size,
                                                    self.num_nodes,
                                                    self.out_features)
        if self.bias is not None:
            return ans + self.bias
        else:
            return ans

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class SparseNodeEncoder(nn.Module):
    def __init__(self, graph, n_hidden, n_embedding,
                 keepprob=1):
        super(SparseNodeEncoder, self).__init__()
        self.num_nodes = graph.number_of_nodes()
        self.n_hidden = n_hidden
        self.n_embedding = n_embedding

        relu = nn.ReLU()
        tanh = nn.Tanh()
        sigmoid = nn.Sigmoid()
        dropout = nn.Dropout(1 - keepprob)

        layers = [SparseNodeLinear(graph,
                                   self.n_hidden[0],
                                   True)]
        for i in range(1, len(n_hidden)):
            layers.append(relu)
            layers.append(dropout)
            layers.append(NodeLinear(self.num_nodes,
                                     self.n_hidden[i - 1], 
                                     self.n_hidden[i]))
        self.encoder = nn.Sequential(*layers, tanh, dropout)
        self.mu = NodeLinear(self.num_nodes,
                             self.n_hidden[-1],
                             self.n_embedding)
        self.var = NodeLinear(self.num_nodes,
                              self.n_hidden[-1],
                              self.n_embedding)

    def forward(self, x):
        x = torch.cat([x[0], x[1]], 2)
        h = self.encoder(x)
        return self.mu(h), self.var(h)