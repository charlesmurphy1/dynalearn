import pickle
import numpy as np
import networkx as nx
import torch
from torch.utils.data.dataset import Dataset



class Markov_Dataset(Dataset):
    def __init__(self,
                 path_to_states,
                 conversion_function=lambda x:x):

        self.conversion_function = conversion_function
        self.complete_data = []

        self.load_data(path_to_states)
        #self.format_states(complete_data)


    def __getitem__(self, index):

        passed_states = self.complete_data[index][0]
        present_states = self.complete_data[index][1]

        return present_states, passed_states


    def __len__(self):
        return len(self.complete_data)


    def load_data(self, path_to_states):


        f = open(path_to_states, "rb")
        idx = 0
        data = pickle.load(f)

        for idx in data:
            if len(data[idx]) > 1:
                d_prev = self.conversion_function(data[idx][0])
                for d in data[idx][1:]:
                    d = self.conversion_function(d)
                    self.complete_data.append((d_prev, d))
                    d_prev = d

        f.close()


class Markov_structured_Dataset(Dataset):
    def __init__(self,
                 path_to_states,
                 path_to_edgelist,
                 conversion_function=lambda x:x):

        self.conversion_function = conversion_function
        self.complete_data = []

        self.g = nx.read_edgelist(path_to_edgelist, nodetype=int)
        self.load_data(path_to_states)


    def __getitem__(self, index):

        passed_states = self.complete_data[index][0]
        present_states = self.complete_data[index][1]

        return present_states, passed_states


    def __len__(self):
        return len(self.complete_data)


    def load_data(self, path_to_states):

        f = open(path_to_states, "rb")
        idx = 0
        data = pickle.load(f)
        neighbors = {n: list(nx.neighbors(self.g, n)) for n in self.g.node}
        for idx in data:
            if len(data[idx]) > 1:
                d_prev = self.conversion_function(data[idx][0])
                for d in data[idx][1:]:
                    d = self.conversion_function(d)
                    d_self = {n:torch.tensor([d[n]]) for n in self.g.node}
                    d_past = {n: torch.cat([d_prev[neighbors[n]],
                                            d_prev[n].view(1)])
                              for n in self.g.node}
                    self.complete_data.append((d_past, d_self))
                    d_prev = d

        f.close()