import pickle
import numpy as np
import networkx as nx
import torch
from torch.utils.data.dataset import Dataset
from dynalearn.dynamics.epidemic import SISNetwork

from dynalearn.utilities import get_bits, add_one_to_bits


class SIS_StateToFloat(object):
    def __call__(self, states):

        present, past = states[0], states[1]

        N = len(present)
        _present = torch.zeros(N)
        _past = torch.zeros(N)
        for i in range(N):
            _present[i] = self._format_state(present[i])
            _past[i] = self._format_state(past[i])

        present = _present.float()
        past = _past.float()

        return present, past

    def __repr__(self):
        return self.__class__.__name__

    def _format_state(self, state):
        if state == 'S':
            return 0
        elif state == 'I':
            return 1
        else:
            raise ValueError('Wrong value of state: got {0} but expected'+
                             ' "S" or "I" for state.'.format(state))


class SIS_ToStructuredData(object):
    def __init__(self, graph=None):
        if graph:
            num_nodes = graph.number_of_nodes()
            adjacency_matrix = torch.from_numpy(nx.to_numpy_array(graph, nodelist=range(num_nodes))).int()
            self.edgeMask = 1 - adjacency_matrix.byte()
        else:
            self.edgeMask = None

    def __call__(self, states):
        
        present, past = states[0], states[1]

        N = len(present_states)
        present_states = present_states
        past_states = past_states

        present_states = present_states.view(1, N)
        past_states = past_states.view(1, N)

        past_states = past_states.repeat(N, 1)

        if self.edgeMask is not None:
            past_states.masked_fill_(self.edgeMask, 0)

        return present_states, past_states


class ResizeData(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, states):
        return states[0].view(self.size), states[1].view(self.size)


class MarkovDataset(Dataset):
    def __init__(self,
                 transform=None):

        self.complete_data = []
        self.transform = transform


    def __getitem__(self, index):
        if self.transform: 
            return self.transform(self.complete_data[index])
        else:
            return self.complete_data[index]


    def __len__(self):
        return len(self.complete_data)


    def load_data(self, path_to_states, pre_transform):

        f = open(path_to_states, "rb")
        idx = 0
        data = pickle.load(f)

        for idx in data:
            if len(data[idx]) > 1:
                d_prev = data[idx][0]
                for d in data[idx][1:]:
                    output = (d, d_prev)
                    if pre_transform:
                        for transform in pre_transform:
                            output = transform(output)
                    self.complete_data.append(output)
                    d_prev = d

        f.close()


class SISDataset(MarkovDataset):
    def __init__(self, rate):
        super(SISDataset, self).__init__()
        self.rate = rate


    def generate(self, graph, num_sample, T,
                 init_active=0.01, dt=0.01,
                 pre_transform=None):
        sis = SISNetwork(graph, self.rate, init_active=0.01, dt=0.01)

        sample = 0

        while sample < num_sample:
            # print(sample, num_sample)
            sis.activity = sis.init_activity()
            sis.continue_simu = True
            t = 0
            prev_activity = None
            while(t < T and sis.continue_simu and sample < num_sample):
                t += dt
                sis.update(record=False)
                sample += 1
                activity = sis.activity.copy()

                if prev_activity is not None:
                    states = (activity, prev_activity)
                    for transform in pre_transform:
                        states = transform(states)
                    self.complete_data.append(states)

                prev_activity = activity


    def init_state(self, graph, init_active=0.01, pre_transform=None):
        sis = SISNetwork(graph, self.rate, init_active=init_active, dt=0.01)
        states = sis.init_active()
        if pre_transform is not None:
            for transform in pre_transform:
                states = transform(states)
        return states


    def get_infected_neighbors(self, graph, state):
        N = graph.number_of_nodes()
        num_infected = np.zeros(N)
        for n in graph.node:
            neighbors = list(graph.neighbors(n))
            num_infected[n] = np.sum(state[neighbors])
        return num_infected


    def get_transition_probability(self, graph, past, dt):
        N = graph.number_of_nodes()
        num_infected = self.get_infected_neighbors(graph, past)

        transprob = np.zeros(N)

        for n in graph.node:
            # infection event
            infection_prob = 1 - (1 - self.rate * dt)**num_infected[n]
            recovery_prob = dt
            if past[n] == 0:
                transprob[n] = infection_prob
            else:
                transprob[n] = 1 - recovery_prob

        return transprob

    def enumerate_all_states(self, graph):
        N = graph.number_of_nodes()
        states = [np.zeros(N)]

        while not all(states[-1] == np.ones(N)):

            new_state = add_one_to_bits(states[-1])
            states.append(new_state)

        return states

        # for 





