import pickle
import numpy as np
import networkx as nx
import torch
from torch.utils.data.dataset import Dataset
from dynalearn.dynamics.epidemic import SISNetwork, SIRNetwork

import dynalearn.utilities as util


class MarkovDataset(Dataset):
    def __init__(self, graph,
                 transform_input=None,
                 transform_graph=None,
                 transform_target=None):

        self.graph = graph
        self.adjacency = torch.tensor(nx.to_numpy_array(graph)).float()
        self.complete_data = []
        self.transform_input = transform_input
        self.transform_graph = transform_graph
        self.transform_target = transform_target


    def __getitem__(self, index):
        input, target = self.complete_data[index]
        adjacency = self.adjacency.clone()

        if self.transform_input: 
            input = self.transform_input(input)
        if self.transform_graph: 
            adjacency = self.transform_graph(adjacency)
        if self.transform_target: 
            target = self.transform_target(target)

        return input.clone(), adjacency.clone(), target.clone()


    def __len__(self):
        return len(self.complete_data)


    def load_data(self, path_to_states, transform_input=None, transform_target=None):

        f = open(path_to_states, "rb")
        idx = 0
        data = pickle.load(f)

        for idx in data:
            if len(data[idx]) > 1:
                input = data[idx][0]
                if transform_input is not None:
                    for t in transform_input:
                        input = t(input)
                for target in data[idx][1:]:
                    if transform_target is not None:
                        for t in transform_target:
                            target = t(target)
                    self.complete_data.append((input, target))
                    d_prev = d

        f.close()


class SISDataset(MarkovDataset):
    def __init__(self, graph, infection_prob, recovery_prob,
                 transform_input=None,
                 transform_target=None,
                 transform_graph=None):
        super(SISDataset, self).__init__(graph,
                                         transform_input,
                                         transform_target,
                                         transform_graph)
        self.infection_prob = infection_prob
        self.recovery_prob = recovery_prob
        self.sis = SISNetwork(self.graph,
                              self.infection_prob,
                              self.recovery_prob)


    def generate(self, num_sample, T, init_param=None, 
                 transform_input=None, transform_target=None,
                 max_null_iter=100, progress_bar=None):

        sample = 0
        self.sis.init_param = init_param
        while sample < num_sample:
            self.sis.states = self.sis.initial_states()
            self.sis.continue_simu = True
            past = self.sis.states.copy()
            null_inter = 0
            
            for t in range(T):

                if progress_bar:
                    progress_bar.update()

                inputs = past.copy()

                if transform_input is not None:
                    for transform in transform_input:
                        inputs = transform(inputs)

                self.sis.update()

                present = self.sis.states.copy()
                target = present.copy()

                if transform_target is not None:
                    for transform in transform_target:
                        target = transform(target)

                self.complete_data.append((inputs, target))

                sample += 1
                past = present
                if not self.sis.continue_simu:
                    null_inter += 1

                if sample == num_sample or null_inter == max_null_iter:
                    break


