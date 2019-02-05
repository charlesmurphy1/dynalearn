import pickle
import numpy as np
import networkx as nx
import torch
from torch.utils.data.dataset import Dataset


class SIS_StateToFloat(object):
    def __call__(self, states):

        present_states, past_states = states[0], states[1]

        N = len(present_states)
        _present = torch.zeros(N)
        _past = torch.zeros(N)
        for i in range(N):
            _present[i] = self._format_state(present_states[i])
            _past[i] = self._format_state(past_states[i])

        present_states = _present.float()
        past_states = _past.float()

        return present_states, past_states

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

    def _format_state(self, state):
        if state == 'S':
            return 0
        elif state == 'I':
            return 1
        else:
            raise ValueError(f'Wrong value of state: got {state} but expected'+
                              ' "S" or "I" for state.')


class SIS_ToStructuredData(object):
    def __init__(self, graph=None):
        if graph:
            num_nodes = graph.number_of_nodes()
            # self_edge = torch.ones(num_nodes, 1).int()
            adjacency_matrix = torch.from_numpy(nx.to_numpy_array(graph)).int()
            self.edge_mask = 1 - adjacency_matrix.byte()
        else:
            self.edge_mask = None

    def __call__(self, states):
        
        present_states, past_states = states[0], states[1]

        N = len(present_states)
        present_states = present_states
        past_states = past_states

        present_states = present_states.view(1, N)
        past_states = past_states.view(1, N)

        past_states = past_states.repeat(N, 1)

        if self.edge_mask is not None:
            past_states.masked_fill_(self.edge_mask, 0)

        return present_states, past_states


class ResizeData(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, states):
        return states[0].view(self.size), states[1].view(self.size)


class Markov_Dataset(Dataset):
    def __init__(self,
                 path_to_states,
                 pre_transform=None,
                 transform=None):

        self.complete_data = []
        self.pre_transform = pre_transform
        self.transform = transform

        self.load_data(path_to_states)


    def __getitem__(self, index):
        if self.transform: 
            return self.transform(self.complete_data[index])
        else:
            return self.complete_data[index]


    def __len__(self):
        return len(self.complete_data)


    def load_data(self, path_to_states):

        f = open(path_to_states, "rb")
        idx = 0
        data = pickle.load(f)

        for idx in data:
            if len(data[idx]) > 1:
                d_prev = data[idx][0]
                for d in data[idx][1:]:
                    output = (d, d_prev)
                    if self.pre_transform:
                        for transform in self.pre_transform:
                            output = transform(output)
                    self.complete_data.append(output)
                    d_prev = d

        f.close()