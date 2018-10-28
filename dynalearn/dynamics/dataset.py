
"""

dataset.py

Created by Charles Murphy on 07-09-18.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines Dynamical_Network_Dataset classes for dynamics on networks.

"""

import torch
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data import Sampler
import networkx as nx
from random import shuffle
import numpy as np

__all__ = ['Node_State_Dataset', 'Dynamical_Network_Dataset',
           'Normal_Dataset', 'Bernoulli_Dataset',
           'Random_Sampler_with_length', 'random_split']

class Node_State_Dataset(Dataset):
    """docstring for Node_State_Dataset"""
    def __init__(self, states, memory=1, conversion_function=lambda x: x):
        super(Node_State_Dataset, self).__init__()
        self.states = conversion_function(states)
        self.T = len(states)
        self.memory = memory


    def __getitem__(self, index):

        present_state = self.states[index]
        passed_states = [None] * self.memory
        time = [None] * self.memory

        for i in range(1, self.memory + 1):
            passed_states[i - 1] = self.states[index - i]

        return present_state, passed_states

    def __len__(self):
        return self.T - memory



class Dynamical_Network_Dataset(Dataset):

    
    def __init__(self,
                 path_to_states,
                 path_to_edgelist=None,
                 memory=1,
                 shuffle_neighbors=False,
                 conversion_function=lambda x:x):

        self.memory = memory
        self.shuffle_neighbors = shuffle_neighbors
        self.conversion_function = conversion_function

        complete_data = self.load_complete_data(path_to_states)
        self.load_network(path_to_edgelist)
        self.load_states(complete_data)

        if path_to_edgelist:
            self.item_function = self._getitem_with_structure
        else:
            self.item_function = self._getitem_without_structure

        return None


    def __getitem__(self, index):
        return self.item_function(index)


    def __len__(self):
        return self.num_node * (self.num_time - self.memory)


    def _getitem_with_structure(self, index):
        node, time = self.node_time_from_index(index)

        present_state, passed_state = self.state_data[node][time]

        neighbors_list = list(self.g.neighbors(node))
        if self.shuffle_neighbors:
            shuffle(neighbors_list)

        neighbors_passed_states = [self.state_data[n][time][1] for n in neighbors_list]

        # return torch.Tensor(present_state), torch.Tensor(passed_state), torch.Tensor(neighbors_passed_states)
        return present_state, passed_state, neighbors_passed_states


    def _getitem_without_structure(self, index):

        present_states = [None] * self.num_node
        passed_states = [None] * self.num_node
        for n in self.g.nodes():
            states = self.state_data[n][index]
            present_states[n] = states[0]
            passed_states[n] = states[1]

        # return torch.Tensor(present_states), torch.Tensor(passed_states)
        return present_states, passed_states


    def node_time_from_index(self, index):

        node_index = self.node_data[index % self.num_node]
        time_index = index // self.num_node + self.memory
        return node_index, time_index


    def load_complete_data(self, path_to_states):

        import pickle
        complete_data = []

        f = open(path_to_states, "rb")
        while True:
            try:
                complete_data.append(pickle.load(f))
            except:
                break
        f.close()

        self.num_node = len(complete_data[0][1])
        self.num_time = len(complete_data)

        return complete_data


    def load_states(self, complete_data):

        # Gathering the nodes' states
        self.state_data = {}
        for i in range(self.num_node):
            node =self.node_data[i]
            states = [None] * self.num_time
            for t in range(self.num_time):
                states[t] = complete_data[t][1][i]

            self.state_data[node] = Node_State_Dataset(states, self.memory,
                                                        self.conversion_function)


        # Gathering the times
        self.time_data = [None] * self.num_time
        for t in range(self.num_time):
            self.time_data[t] = complete_data[t][0]

        return None


    def load_network(self, path_to_edgelist=None):

        # Gathering graph
        if path_to_edgelist is not None:
            self.g = nx.read_edgelist(path_to_edgelist)
        else:
            self.g = nx.empty_graph(self.num_node)

        # Gathering nodes and indexes
        self.node_data = {}
        for i, node in enumerate(self.g.nodes()):
            self.node_data[i] = node

        return None


class Normal_Dataset(Dataset):
    """docstring for Normal_Dataset"""
    def __init__(self, numsample=1000, dim=1, means=None, stds=None):
        self.dim = dim
        if means is None:
            self.means = np.random.normal(0, 1, self.dim)
        else:
            self.means = means

        if stds is None:
            self.stds = abs(1 + np.random.normal(loc=0, scale=1, size=self.dim))
        else:
            self.stds = stds

        self.data = []
        self._generate_data(numsample)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _generate_data(self, numsample):
        for i in range(numsample):
            self.data.append(np.random.normal(loc=self.means, scale=self.stds))

        return None


class Bernoulli_Dataset(Dataset):
    """docstring for Bernoulli_Dataset"""
    def __init__(self, numsample=1000, dim=1, p=None):
        self.dim = dim
        if p is None:
            self.p = np.random.rand(self.dim)
        else:
            self.p = p

        self.data = []
        self._generate_data(numsample)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _generate_data(self, numsample):
        for i in range(numsample):
            x = np.zeros(self.dim)
            r = np.random.rand(self.dim)
            x[r < self.p] = 1
            x = torch.Tensor(x)
            self.data.append(x)

        return None


class Random_Sampler_with_length(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, length):
        self.length = length
        self.data_source = data_source

    def __iter__(self):
        return iter(torch.randint(0, len(self.data_source), self.length).tolist())

    def __len__(self):
        return self.length


def random_split(dataset, val_size):
    index = set(range(len(dataset)))
    val_index = list(np.random.choice(list(index), int(val_size * len(dataset))))
    train_index = list(index.difference(set(val_index)))

    return Subset(dataset, train_index), Subset(dataset, val_index)








# if __name__ == '__main__':
#     N = 10
#     T = 5
#     path_to_states = "testdata/SIS_states.b"
#     path_to_edgelist = "testdata/SIS_net.txt"
#     memory = 2
#     shuffle_neighbors = False

#     def conv_func_SIS(x):
#         for i, val in enumerate(x):
#             if val=="S": x[i] = [0]
#             if val=="I": x[i] = [1]
            
#         return x

#     dataset = Dynamical_Network_Dataset(path_to_states, path_to_edgelist,
#                                       memory, shuffle_neighbors,
#                                       conv_func_SIS)

#     print("Complete dataset")
#     c_d = dataset.load_complete_data(path_to_states)
#     string = ""
#     for d in c_d:
#         string += str(d)
#         string += "\n"

#     print(string)

#     print("Basic format", dataset[0])

#     for i in range(N * (T - memory)):
#         n, t = dataset.node_time_from_index(i)
#         string = "Node {0}, time {1}, present {2}, passed {3}".format(n, t, dataset[i][0], dataset[i][1])
#         string += " Neighbors' states "

#         for j, d in enumerate(dataset[i][2]):
#             string += "{0} ".format(d)

#         print(string)

if __name__ == '__main__':

    from random import choice

    dataset = Bernoulli_Dataset(10000, 10)

    train, val = random_split(dataset, 0.1)

    train_loader = torch.utils.data.DataLoader(train, 
                                             batch_size=2,
                                             shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, 
                                             batch_size=2,
                                             shuffle=True)
    print(len(train), choice(list(val_loader)))