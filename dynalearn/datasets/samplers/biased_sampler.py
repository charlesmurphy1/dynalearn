from .random_sampler import RandomSampler
import numpy as np
from time import time
import tqdm
from abc import abstractmethod


class BiasedSampler(RandomSampler):
    def __init__(self, name, config, verbose=0):
        super(BiasedSampler, self).__init__(name, config, verbose)
        self.sampling_bias = config.sampling_bias

    def update_weights(self, graphs, inputs):
        total_num_samples = np.sum([inputs[g].shape[0] for g in graphs])

        if self.verbose == 1:
            p_bar = tqdm.tqdm(range(total_num_samples))

        # Update dist
        dist = dict()
        for g in graphs:
            adj = graphs[g]
            for s in inputs[g]:
                t0 = time()
                summaries = self.summarize(adj, s)
                for sum in summaries:
                    if sum in dist:
                        dist[sum] += 1 / total_num_samples
                    else:
                        dist[sum] = 1 / total_num_samples
                t1 = time()
                if self.verbose == 1:
                    p_bar.set_description(
                        "Update dist - " + str(round(t1 - t0, 5)) + "s"
                    )
                    p_bar.update()

        if self.verbose == 1:
            p_bar.close()
            p_bar = tqdm.tqdm(range(total_num_samples))

        self.node_weights = dict()
        self.state_weights = dict()
        self.graph_weights = dict()
        for g in graphs:
            adj = graphs[g]
            self.node_weights[g] = dict()
            self.state_weights[g] = dict()
            for t, s in enumerate(inputs[g]):
                t0 = time()
                summaries = self.summarize(adj, s)
                self.node_weights[g][t] = np.zeros(self.num_nodes[g])
                for n, sum in zip(self.avail_node_set[g][t], summaries):
                    self.node_weights[g][t][n] = dist[sum] ** (-1)
                self.state_weights[g][t] = np.sum(self.node_weights[g][t])
                t1 = time()
                if self.verbose == 1:
                    p_bar.set_description(
                        "Update weights - " + str(round(t1 - t0, 5)) + "s"
                    )
                    p_bar.update()
            self.graph_weights[g] = np.sum(
                [self.state_weights[g][t] for t in self.state_set[g]]
            )

    @abstractmethod
    def summarize(self, adj, state):
        raise NotImplementedError()


class DegreeBiasedSampler(BiasedSampler):
    def __init__(self, name, config, verbose=0):
        super(DegreeBiasedSampler, self).__init__(name, config, verbose)

    def summarize(self, adj, state):
        return np.sum(adj, axis=0)


class StateBiasedSampler(BiasedSampler):
    def __init__(self, name, config, verbose=0):
        super(StateBiasedSampler, self).__init__(name, config, verbose)
        self.dynamics_states = config.dynamics_states

    def summarize(self, adj, state):
        summaries_arr = np.zeros((state.shape[0], len(self.dynamics_states) + 1))
        summaries_arr[:, 0] = state
        for s in self.dynamics_states:
            summaries_arr[:, int(s) + 1] = np.matmul(adj, state == s)

        summaries = [tuple(s) for s in summaries_arr]
        return summaries
