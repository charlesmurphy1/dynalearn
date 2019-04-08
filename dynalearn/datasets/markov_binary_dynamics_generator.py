import numpy as np
import networkx as nx
import random
from dynalearn.dynamics.epidemic import SISNetwork
import time


class MarkovBinaryDynamicsGenerator():
    def __init__(self, graph_gen, dynamics_gen, batch_size,
                 shuffle=False, max_null_iter=100):
        self.graph_gen = graph_gen
        self.dynamics_gen = dynamics_gen

        # Data
        self.graph_inputs = dict()
        self.state_inputs = dict()
        self.targets = dict()
        self.sample_weights = dict()
        self.state_index = dict()
        self.graph_index = set()
    
        # Params        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_null_iter = max_null_iter
        self.with_structure = with_structure
        self.gamma = gamma
        self.iteration = 0
        self.current_graph_name = None

        if type(prohibited_node_index) is not list:
            raise ValueError('prohibited_node_index must be a list')
        else:
            self.prohibited_node_index = prohibited_node_index

    def _reset_graph_index(self):
        self.graph_index = set(self.graph_inputs.keys())

    def sample(self):
        # if self.shuffle:
        #     index =  random.sample(self.state_index, 1)[0]
        #     graph_name = random.sample(self.graph_index, 1)[0]
        # else:
        #     if len(self.state_index) == 0:
        #         if len(self.graph_index) == 0:
        #             self.graph_index = set(self.graph_inputs.keys())

        #         self.current_graph_name = self.graph_index.pop()
        #         self.state_index = set(range(self.state_inputs[self.current_graph_name].shape[0]))
        #     index = self.state_index.pop()
        #     graph_name = self.current_graph_name

        # self.state_index = self.state_index.difference([index])
        # if len(self.state_index) == 0:
        #     self.state_index = set(range(self.data_input.shape[0]))

        # if self.batch_size is None:
        #     weights = np.ones(self.N)
        # else:
        #     node_index = np.random.choice(range(N),
        #                                   replace=False,
        #                                   size=self.batch_size,
        #                                   p=sample_weights)
        #     weights = np.zeros(self.N)
        #     weights[node_index] += 1
        # inputs = self.data_input[index, :]
        # targets = self.data_target[index, :]
        # return inputs, targets, weights
        if self.shuffle:
            if len(self.graph_index) == 0:
                self.reset_graph_index()
            graph_name = random.sample(self.graph_index, 1)[0]


    def generate(self, num_sample, T, progress_bar=None, gamma=0.):

        if progress_bar: bar = progress_bar(range(num_sample))

        sample = 0
        name, graph = self.graph_gen.generate()
        N = graph.number_of_nodes()


        self.graph_inputs[name] = graph
        self.state_inputs[name] = np.zeros([num_sample, N])
        self.targets[name] = np.zeros([num_sample, N])
        self.sample_weights[name] = self.get_sample_weights(graph, gamma)
        self.dynamics.graph = graph

        if self.current_graph_name is None:
            self.current_graph_name = name

        while sample < num_sample:
            self.dynamics.initialize_states()
            null_iteration = 0
            for t in range(T):

                t0 = time.time()
                inputs, targets = self.update_dynamics()
                self.state_inputs[name][sample, :] = inputs
                self.target[name][sample, :] = targets
                t1 = time.time()

                if progress_bar:
                    bar.set_description(str(round(t1 - t0, 5)))
                    bar.update()

                sample += 1
                if not self.dynamics.continue_simu:
                    null_iteration += 1

                if sample == num_sample or null_iteration == self.max_null_iter:
                    break

        self.state_index[name] = set(range(self.state_inputs[name].shape[0]))
        self.graph_index.add(name)


    def get_sample_weights(self, graph, gamma):
        degrees = dict(graph.degree())
        deg_seq = np.array([degrees[i] for i in degrees])
        return deg_seq**gamma np.sum(deg_seq**gamma)


    def __len__(self):

        return np.prod([self.state_inputs[name].shape[0] for name in self.state_inputs])


    def __iter__(self):
        return self


    def update_dynamics(self):
        inputs = self.dynamics.states
        self.dynamics.update()
        targets = self.dynamics.states
        return inputs, targets
        

    def __next__(self):
        inputs, targets, weights = self.sample()

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







