import numpy as np
import networkx as nx
import random
from dynalearn.dynamics.epidemic import SISNetwork
import time


class MarkovBinaryDynamicsGenerator():
    def __init__(self, batch_size, shuffle=False,
                 max_null_iter=100, with_structure=False,
                 prohibited_node_index=[], gamma=0):
        self.data_input = None
        self.data_target = None
        self.graph = None
        self.dynamics = None
        
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.max_null_iter = max_null_iter
        self.with_structure = with_structure
        self.gamma = gamma
        self.iteration = 0
        self.avail_index = set()

        if type(prohibited_node_index) is not list:
            raise ValueError('prohibited_node_index must be a list')
        else:
            self.prohibited_node_index = prohibited_node_index

        if self.with_structure:
            self.sample = self.with_structure_sampling
        else:
            self.sample = self.without_structure_sampling


    def sample(self):
        raise NotImplementedError()


    def generate(self, num_sample, T, progress_bar=None):
        sample = 0

        if progress_bar: bar = progress_bar(range(num_sample))
        self.data_input = np.zeros([num_sample, self.N])
        self.data_target = np.zeros([num_sample, self.N])

        while sample < num_sample:
            self.dynamics.initialize_states()
            null_iteration = 0
            for t in range(T):

                t0 = time.time()
                inputs, targets = self.update_dynamics()
                self.data_input[sample, :] = inputs
                self.data_target[sample, :] = targets
                t1 = time.time()

                if progress_bar:
                    bar.set_description(str(round(t1 - t0, 5)))
                    bar.update()

                sample += 1
                if not self.dynamics.continue_simu:
                    null_iteration += 1

                if sample == num_sample or null_iteration == self.max_null_iter:
                    break
        self.avail_index = set(range(self.data_input.shape[0]))


    def set_graph(self, graph):
        self.graph = graph
        self.N = self.graph.number_of_nodes()
        self.adj = nx.to_numpy_array(self.graph) + np.eye(self.N)
        degrees = dict(graph.degree())
        deg_seq = np.array([degrees[i] for i in degrees])
        s_w = deg_seq**self.gamma
        self.sample_weights = s_w * 1
        self.sample_weights[self.prohibited_node_index] = 0
        self.sample_weights /= np.sum(self.sample_weights)

    def set_gamma(self, gamma):
        self.gamma = gamma
        degrees = dict(self.graph.degree())
        deg_seq = np.array([degrees[i] for i in degrees])
        s_w = deg_seq**self.gamma
        self.sample_weights = s_w * 1
        self.sample_weights[self.prohibited_node_index] = 0
        self.sample_weights /= np.sum(self.sample_weights)

    def clear(self):
        self.data_input = None
        self.data_target = None
        self.graph = None

    def __len__(self):
        if self.data_input is not None:
            return self.data_input.shape[0]
        else:
            return 0

    def __iter__(self):
        if self.dynamics is None:
            raise ValueError("self.data must always be defined.")
        elif self.graph is None:
            raise ValueError("self.graph must always not be None.")
        elif self.data_input is None and not self.online:
            raise ValueError("data must not be empty.")
        else:
            return self


    def update_dynamics(self):
        inputs = self.dynamics.states
        self.dynamics.update()
        targets = self.dynamics.states
        return inputs, targets


    def with_structure_sampling(self):
        if self.shuffle:
            index =  random.sample(self.avail_index, 1)[0]
        else:
            index = self.iteration
            self.iteration += 1
            if self.iteration == self.data_input.shape[0]:
                self.iteration = 0

        self.avail_index = self.avail_index.difference([index])
        if len(self.avail_index) == 0:
            self.avail_index = set(range(self.data_input.shape[0]))

        if self.batch_size is None:
            weights = np.ones(self.N)
        else:
            node_index = np.random.choice(range(N),
                                          replace=False,
                                          size=self.batch_size,
                                          p=self.sample_weights)
            weights = np.zeros(self.N)
            weights[node_index] += 1
        inputs = self.data_input[index, :]
        targets = self.data_target[index, :]
        return inputs, targets, weights


    def without_structure_sampling(self):
        if len(self.avail_index) < self.batch_size:
            batch_size = len(self.avail_index)
        else:
            batch_size = self.batch_size

        if self.shuffle:
            index =  random.sample(self.avail_index, batch_size)
        else:
            index = self.avail_index[:batch_size]

        self.avail_index = self.avail_index.difference(index)
        if len(self.avail_index) == 0:
            self.avail_index = set(range(self.data_input.shape[0]))

        inputs = self.data_input[index, :]
        targets = self.data_target[index, :]
        weights = np.ones(self.batch_size)

        return inputs, targets, weights
        

    def __next__(self):
        inputs, targets, weights = self.sample()

        if self.with_structure:
            adj = self.adj
            inputs = [inputs, adj]

        return inputs, targets, weights


class SISGenerator(MarkovBinaryDynamicsGenerator):
    def __init__(self, graph, infection_prob, recovery_prob, batch_size,
                 shuffle=False, init_param=None, max_null_iter=100,
                 with_structure=False,  prohibited_node_index=None, gamma=0):
        super(SISGenerator, self).__init__(batch_size,
                                           shuffle=shuffle,
                                           max_null_iter=max_null_iter,
                                           with_structure=with_structure,
                                           gamma=gamma,
                                           prohibited_node_index=prohibited_node_index)
        self.set_graph(graph)
        self.infection_prob = infection_prob
        self.recovery_prob = recovery_prob
        self.init_param = init_param
        self.max_null_iter = max_null_iter

        self.dynamics = SISNetwork(self.graph,
                                   self.infection_prob,
                                   self.recovery_prob,
                                   self.init_param)







