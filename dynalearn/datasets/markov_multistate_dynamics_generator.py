import numpy as np
import networkx as nx
import random
from dynalearn.dynamics.epidemic import SISNetwork
import time


class MarkovMultiStateDynamicsGenerator():
    def __init__(self, batch_size, online=False, shuffle=False,
                  max_null_iter=100, with_structure=False, n_states=1):
        self.data_input = None
        self.data_target = None
        self.graph = None
        self.dynamics = None
        self.n_states = n_states
        
        self.batch_size = batch_size

        self.online = online
        self.shuffle = shuffle
        self.max_null_iter = max_null_iter
        self.with_structure = with_structure
        self.iteration = 0

        if self.batch_size is None:
            if self.online:
                self.sample = self.without_batch_online_sampling
            else:
                self.sample = self.without_batch_offline_sampling
        else:
            if self.online:
                self.sample = self.with_batch_online_sampling
            else:
                self.sample = self.with_batch_offline_sampling

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

    def __next__(self):
        inputs, targets = self.sample()

        if self.with_structure:
            if self.batch_size is None:
                adj = self.adj
            else:
                bs = inputs.shape[0]
                adj = np.repeat(self.adj.reshape(1, self.N, self.N), bs,axis=0)
            inputs = [inputs, adj]

        return inputs, targets

    def sample(self):
        raise NotImplemented

    def to_one_hot(self, val):
        arr = np.zeros(self.n_states)
        arr[int(val)] = 1
        return arr

    def update_dynamics(self):
        inputs = np.array([self.to_one_hot(self.dynamics.activity[v])
                           for v in self.graph.node])
        self.dynamics.update()
        targets = np.array([self.to_one_hot(self.dynamics.activity[v])
                            for v in self.graph.node])
        return inputs, targets


    def generate(self, num_sample, T, progress_bar=None):
        sample = 0

        if progress_bar: bar = progress_bar(range(num_sample))
        self.data_input = np.zeros([num_sample, self.N, self.n_states])
        self.data_target = np.zeros([num_sample, self.N, self.n_states])

        while sample < num_sample:
            self.dynamics.activity = self.dynamics.init_activity()
            self.dynamics.continue_simu = True
            null_iteration = 0
            for t in range(T):            
                t0 = time.time()
                inputs, targets = self.update_dynamics()
                self.data_input[sample, :, :] = inputs
                self.data_target[sample, :, :] = targets
                t1 = time.time()

                if progress_bar:
                    bar.set_description(str(round(t1-t0, 5)))
                    bar.update()

                sample += 1
                if not self.dynamics.continue_simu:
                    null_iteration += 1

                if sample == num_sample or null_iteration == self.max_null_iter:
                    break

    def set_graph(self, graph):
        self.graph = graph
        self.N = self.graph.number_of_nodes()
        self.adj = nx.to_numpy_array(self.graph) + np.eye(self.N)

    def clear(self):
        self.data_input = None
        self.data_target = None
        self.graph = None


    def without_batch_online_sampling(self):
        self.dynamics.activity = self.dynamics.init_activity()
        return self.update_dynamics()

    def with_batch_online_sampling(self):
        inputs = []
        targets = []
        for i in range(self.batch_size):
            self.dynamics.activity = self.dynamics.init_activity()
            input, target = self.update_dynamics()
            inputs.append(input)
            targets.append(target)

        return inputs, targets


    def without_batch_offline_sampling(self):
        if self.shuffle:
            index =  random.choice(range(self.data_input.shape[0]))
        else:
            index = self.iteration
            self.iteration += 1
            if self.iteration == self.data_input.shape[0]:
                self.iteration = 0
        inputs = self.data_input[index, :, :]
        targets = self.data_target[index, :, :]
        return inputs, targets


    def with_batch_offline_sampling(self):
        if self.shuffle:
            index =  random.sample(range(self.data_input.shape[0]),
                                   self.batch_size)
        else:
            i_prev = self.iteration * self.batch_size
            i_next = (self.iteration + 1) * self.batch_size
            self.iteration += 1
            if i_next > self.data_input.shape[0]:
                i_next = -1
                self.iteration = 0
            index = range(self.data_input.shape[0])[i_prev:i_next]

        inputs = self.data_input[index, :, :]
        targets = self.data_target[index, :, :]
        return inputs, targets


class SISMultiStateGenerator(MarkovMultiStateDynamicsGenerator):
    def __init__(self, graph, infection_prob, recovery_prob, batch_size,
                 online=False, shuffle=False, init_active=None, 
                 max_null_iter=100, with_structure=False, init_param=None):
        super(SISMultiStateGenerator, self).__init__(batch_size,
                                           online=online,
                                           shuffle=shuffle,
                                           max_null_iter=max_null_iter,
                                           with_structure=with_structure,
                                           n_states=2)
        self.set_graph(graph)
        self.infection_prob = infection_prob
        self.recovery_prob = recovery_prob
        self.init_active = init_active
        self.max_null_iter = max_null_iter

        self.dynamics = SISNetwork(self.graph,
                                   self.infection_prob,
                                   self.recovery_prob,
                                   init_param)







