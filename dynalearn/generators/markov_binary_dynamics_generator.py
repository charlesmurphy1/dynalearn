import numpy as np
import networkx as nx
import random
from dynalearn.generators.dynamics.epidemic import SISDynamics
import time


class MarkovBinaryDynamicsGenerator():
    def __init__(self, graph_gen, state_gen, batch_size,
                 shuffle=False, prohibited_node_index=[],
                 max_null_iter=100):
        self.graph_gen = graph_gen
        self.state_gen = state_gen
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_graph_name = None
        self.prohibited_node_index = prohibited_node_index
        self.max_null_iter = max_null_iter

        # Data
        self.graph_inputs = dict()
        self.state_inputs = dict()
        self.targets = dict()
        self.sample_weights = dict()

        # Iterations
        self.state_index = dict()
        self.graph_index = set()


    def generate(self, num_sample, T, progress_bar=None, gamma=0.):

        if progress_bar: bar = progress_bar(range(num_sample))

        sample = 0
        name, graph = self.graph_gen.generate()
        N = graph.number_of_nodes()


        self.graph_inputs[name] = nx.to_numpy_array(graph)
        self.state_inputs[name] = np.zeros([num_sample, N])
        self.targets[name] = np.zeros([num_sample, N])
        self.sample_weights[name] = self._get_sample_weights(graph, gamma)
        self.state_gen.graph = graph

        while sample < num_sample:
            self.state_gen.initialize_states()
            null_iteration = 0
            for t in range(T):

                t0 = time.time()
                inputs, targets = self._update_states()
                self.state_inputs[name][sample, :] = inputs
                self.targets[name][sample, :] = targets
                t1 = time.time()

                if progress_bar:
                    bar.set_description(str(round(t1 - t0, 5)))
                    bar.update()

                sample += 1
                if not self.state_gen.continue_simu:
                    null_iteration += 1

                if sample == num_sample or null_iteration == self.max_null_iter:
                    break

        self.state_index[name] = set(range(self.state_inputs[name].shape[0]))
        self.graph_index.add(name)
        if self.current_graph_name is None:
            self.current_graph_name = name


    def _reset_graph_index(self):
        self.graph_index = set(self.graph_inputs.keys())


    def _reset_state_index(self, graph_name):
        n_sample = self.state_inputs[graph_name].shape[0]
        self.state_index[graph_name] = set(range(n_sample))


    def _sample(self):
        if self.shuffle:
            g_index = random.sample(self.graph_index, 1)[0]
            s_index = random.sample(self.state_index[g_index], 1)[0]
            self.state_index[g_index].remove(s_index)
        else:
            g_index = self.current_graph_name
            s_index = self.state_index[g_index].pop()

        if len(self.state_index[g_index]) == 0:
            self.graph_index.remove(g_index)
            if len(self.graph_index) == 0:
                self._reset_graph_index()
                for g in self.graph_index: self._reset_state_index(g)
            self.current_graph_name = list(self.graph_index)[0]

        adj = self.graph_inputs[g_index]
        inputs = self.state_inputs[g_index][s_index, :]
        targets = self.targets[g_index][s_index, :]
        weights = self.sample_weights[g_index]
        return [inputs, adj], targets, weights


    def _get_sample_weights(self, graph, gamma):
        degrees = dict(graph.degree())
        deg_seq = np.array([degrees[i] for i in degrees])
        deg_seq[self.prohibited_node_index] = 0
        return deg_seq**gamma / np.sum(deg_seq**gamma)


    def __len__(self):
        names = self.state_inputs.keys()
        return np.sum([self.state_inputs[n].shape[0] for n in names])


    def __iter__(self):
        return self


    def _update_states(self):
        inputs = self.state_gen.states
        self.state_gen.update()
        targets = self.state_gen.states
        return inputs, targets
        

    def __next__(self):
        return self._sample()


class SISGenerator(MarkovBinaryDynamicsGenerator):
    def __init__(self, graph_gen, infection_prob, recovery_prob, batch_size,
                 shuffle=False, init_param=None, max_null_iter=100,
                 prohibited_node_index=[]):

        self.infection_prob = infection_prob
        self.recovery_prob = recovery_prob
        self.init_param = init_param

        state_gen = SISDynamics(self.infection_prob,
                               self.recovery_prob,
                               self.init_param)

        super(SISGenerator, self).__init__(graph_gen, state_gen,
                                           batch_size,
                                           shuffle=shuffle,
                                           prohibited_node_index=prohibited_node_index,
                                           max_null_iter=max_null_iter)






