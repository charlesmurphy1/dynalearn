import math
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter


class CompleteEncoder(nn.Module):
    def __init__(self, graph, n_hidden, n_embedding, keepprob=1):
        super(CompleteEncoder, self).__init__()
        if type(n_hidden) == int: n_hidden = [n_hidden]
    
        self.num_nodes = graph.number_of_nodes()
        self.n_hidden = n_hidden
        self.n_embedding = n_embedding

        # Functions
        relu = nn.ReLU()
        tanh = nn.Tanh()
        sigmoid = nn.Sigmoid()
        dropout = nn.Dropout(1 - keepprob)
        
        # Encoder: Inference network weights
        layers = [nn.Linear(2 * self.num_nodes, self.n_hidden[0])]
        for i in range(1, len(self.n_hidden)):
            batchnorm = nn.BatchNorm1d(self.n_hidden[i - 1])
            layers.append(batchnorm)
            layers.append(relu)
            layers.append(dropout)
            layers.append(nn.Linear(self.n_hidden[i - 1], self.n_hidden[i]))
        batchnorm = nn.BatchNorm1d(self.n_hidden[i])
        self.encoder = nn.Sequential(*layers, batchnorm, relu, dropout)
        # self.encoder = nn.Sequential(*layers, relu, dropout)
        
        # Embedding networks weights
        self.mu = nn.Linear(self.n_hidden[-1],
                            self.n_embedding)
        self.var = nn.Linear(self.n_hidden[-1],
                             self.n_embedding)

    def forward(self, x, c):
        x = torch.cat([x, c], 1)
        h = self.encoder(x)
        return self.mu(h), self.var(h)


class CompleteDecoder(nn.Module):
    def __init__(self, graph, n_hidden, n_embedding, keepprob=1):
        super(CompleteDecoder, self).__init__()
        if type(n_hidden) == int: n_hidden = [n_hidden]

        self.num_nodes = graph.number_of_nodes()
        self.n_hidden = n_hidden
        self.n_embedding = n_embedding

        # Functions
        relu = nn.ReLU()
        tanh = nn.Tanh()
        sigmoid = nn.Sigmoid()
        dropout = nn.Dropout(1 - keepprob)
    
        # Decoder: Generative network weights
        layers = []
        for i in range(len(self.n_hidden) - 1, 0, -1):

            layers.append(nn.Linear(self.n_hidden[i],
                                    self.n_hidden[i - 1]))
            batchnorm = nn.BatchNorm1d(self.n_hidden[i - 1])
            layers.append(batchnorm)
            layers.append(relu)
            layers.append(dropout)
        self.decoder = nn.Sequential(nn.Linear(self.n_embedding + self.num_nodes,
                                               self.n_hidden[-1]),
                                      nn.BatchNorm1d(self.n_hidden[-1]),
                                      relu,
                                      dropout,
                                      *layers,
                                      nn.Linear(self.n_hidden[0],
                                                self.num_nodes),
                                      sigmoid)


    def forward(self, z, c):
        z = torch.cat([z, c], 1)
        return self.decoder(z)

class NodeLinear(nn.Module):
    def __init__(self, in_features, out_features, num_nodes, bias=True):
        super(NodeLinear, self).__init__()
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.Tensor(num_nodes,
                                             in_features,
                                             out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features,
                                               num_nodes))
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
        input = input.permute(0, 2, 1)
        input = input.view(batch_size, self.num_nodes, 1, self.in_features)
        ans = torch.matmul(input, self.weight).view(batch_size,
                                                    self.num_nodes,
                                                    self.out_features)
        ans = ans.permute(0, 2, 1)
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

        layers = [NodeLinear(self.num_nodes + 1,
                                  self.n_hidden[0],
                                  self.num_nodes)]
        for i in range(1, len(n_hidden)):
            batchnorm = nn.BatchNorm1d(self.n_hidden[i - 1])
            layers.append(batchnorm)
            layers.append(relu)
            layers.append(dropout)
            layers.append(NodeLinear(self.n_hidden[i - 1], 
                                     self.n_hidden[i],
                                     self.num_nodes))
        batchnorm = nn.BatchNorm1d(self.n_hidden[-1])
        self.encoder = nn.Sequential(*layers,  batchnorm, relu, dropout)
        self.mu = NodeLinear(self.n_hidden[-1],
                             self.n_embedding,
                             self.num_nodes)
        self.var = NodeLinear(self.n_hidden[-1],
                              self.n_embedding,
                              self.num_nodes)


    def forward(self, x, c):
        x = torch.cat([x, c], 1)
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
            layers.append(NodeLinear(self.n_hidden[i],
                                     self.n_hidden[i - 1],
                                     self.num_nodes))
            batchnorm = nn.BatchNorm1d(self.n_hidden[i - 1])
            layers.append(batchnorm)
            layers.append(relu)
            layers.append(dropout)
        self.decoder = nn.Sequential(NodeLinear(self.n_embedding + self.num_nodes,
                                                self.n_hidden[-1],
                                                self.num_nodes),
                                     nn.BatchNorm1d(self.n_hidden[-1]),
                                     relu,
                                     dropout,
                                     *layers,
                                     NodeLinear(self.n_hidden[0],
                                                1,
                                                self.num_nodes),
                                     sigmoid)


    def forward(self, z, c):
        z = torch.cat([z, c], 1)
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
                                               1,
                                               self.num_nodes + 1),
                                  requires_grad=False)

        self.edgeMask.data = self.compute_edgeMask(graph)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features,
                                               self.num_nodes))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def compute_edgeMask(self, graph):
        adj = nx.to_numpy_array(graph, nodelist=range(self.num_nodes))
        np.fill_diagonal(adj, 1)
        adj = np.concatenate([np.ones([self.num_nodes, 1]), adj], 1)
        mask = 1 - torch.tensor(adj).byte()
        return mask.view(self.num_nodes, 1, self.num_nodes + 1)


    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    def forward(self, input):
        batch_size = input.size(0)
        input = input.permute(0, 2, 1)
        input = input.view(batch_size, self.num_nodes, 1, self.num_nodes + 1)
        input.masked_fill_(self.edgeMask, 0)
        ans = torch.matmul(input, self.weight).view(batch_size,
                                                    self.num_nodes,
                                                    self.out_features)
        ans = ans.permute(0, 2, 1)
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

        layers = [SparseNodeLinear(graph, self.n_hidden[0], True)]
        for i in range(1, len(n_hidden)):
            batchnorm = nn.BatchNorm1d(self.n_hidden[i - 1])
            layers.append(batchnorm)
            layers.append(relu)
            layers.append(dropout)
            layers.append(NodeLinear(self.n_hidden[i - 1], 
                                     self.n_hidden[i],
                                     self.num_nodes))
        batchnorm = nn.BatchNorm1d(self.n_hidden[-1])
        self.encoder = nn.Sequential(*layers, batchnorm, relu, dropout)
        self.mu = NodeLinear(self.n_hidden[-1],
                             self.n_embedding,
                             self.num_nodes)
        self.var = NodeLinear(self.n_hidden[-1],
                              self.n_embedding,
                              self.num_nodes)

    def forward(self, x, c):
        x = torch.cat([x, c], 1)
        h = self.encoder(x)
        return self.mu(h), self.var(h)


class NodeDegreeEncoder(nn.Module):
    def __init__(self, graph, n_hidden, n_embedding, kmax=None, keepprob=1):
        super(NodeDegreeEncoder, self).__init__()
        self.num_nodes = graph.number_of_nodes()
        if kmax is None:
            self.kmax = max(dict(graph.degree()).values())
        else:
            self.kmax = kmax
        self.n_hidden = n_hidden
        self.n_embedding = n_embedding

        relu = nn.ReLU()
        tanh = nn.Tanh()
        sigmoid = nn.Sigmoid()
        dropout = nn.Dropout(1 - keepprob)

        layers = [nn.Conv1d(self.kmax + 3, n_hidden[0], 1)]
        for i in range(1, len(n_hidden)):
            batchnorm = nn.BatchNorm1d(self.n_hidden[i - 1])
            layers.append(batchnorm)
            layers.append(relu)
            layers.append(dropout)
            layers.append(nn.Conv1d(self.n_hidden[i - 1], self.n_hidden[i], 1))
        batchnorm = nn.BatchNorm1d(self.n_hidden[-1])
        self.encoder = nn.Sequential(*layers, batchnorm, relu, dropout)
        self.mu = nn.Conv1d(self.n_hidden[-1], self.n_embedding, 1)
        self.var = nn.Conv1d(self.n_hidden[-1], self.n_embedding, 1)


    def forward(self, x, c):
        x = torch.cat([x, c], 1)
        h = self.encoder(x)
        return self.mu(h), self.var(h)


class NodeDegreeDecoder(nn.Module):
    def __init__(self, graph, n_hidden, n_embedding, kmax=None, keepprob=1):
        super(NodeDegreeDecoder, self).__init__()
        self.num_nodes = graph.number_of_nodes()
        if kmax is None:
            self.kmax = max(dict(graph.degree()).values())
        else:
            self.kmax = kmax
        self.n_hidden = n_hidden
        self.n_embedding = n_embedding

        relu = nn.ReLU()
        tanh = nn.Tanh()
        sigmoid = nn.Sigmoid()
        dropout = nn.Dropout(1 - keepprob)

        layers = []
        for i in range(len(n_hidden) - 1, 0, -1):
            layers.append(nn.Conv1d(self.n_hidden[i], self.n_hidden[i - 1], 1))
            batchnorm = nn.BatchNorm1d(self.n_hidden[i - 1])
            layers.append(batchnorm)
            layers.append(relu)
            layers.append(dropout)
        self.decoder = nn.Sequential(nn.Conv1d(self.n_embedding + self.kmax + 2,
                                               self.n_hidden[-1], 1),
                                     nn.BatchNorm1d(self.n_hidden[-1]),
                                     relu,
                                     dropout,
                                     *layers,
                                     nn.Conv1d(self.n_hidden[0], 1, 1),
                                     sigmoid)

    def forward(self, z, c):
        z = torch.cat([z, c], 1)
        return self.decoder(z)