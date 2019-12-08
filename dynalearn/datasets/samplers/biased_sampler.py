from .random_sampler import RandomSampler
import numpy as np
from time import time
import tqdm
from abc import abstractmethod


class BiasedSampler(RandomSampler):
    def __init__(self, name, config, verbose=0):
        self.sampling_bias = config.sampling_bias
        super(BiasedSampler, self).__init__(name, config, verbose)

    def update_weights(self, graphs, inputs):
        if self.verbose:
            nn = np.sum([inputs[n].shape[0] for n in graphs])
            p_bar = tqdm.tqdm(range(nn))

        # Update counts
        counts = dict()
        for n in graphs:
            adj = graphs[n]
            for s in inputs[n]:
                t0 = time()
                summaries = self.summarize(adj, s)
                for sum in summaries:
                    if sum in counts:
                        counts[sum] += 1
                    else:
                        counts[sum] = 1
                t1 = time()
                if self.verbose:
                    p_bar.set_description(
                        "Update counts - " + str(round(t1 - t0, 5)) + "s"
                    )
                    p_bar.update()
        if self.verbose:
            p_bar.close()
            nn = np.sum([inputs[n].shape[0] for n in graphs])
            p_bar = tqdm.tqdm(range(nn))

        self.node_weights = dict()
        self.state_weights = dict()
        self.graph_weights = dict()
        for n in graphs:
            adj = graphs[n]
            self.node_weights[n] = dict()
            self.state_weights[n] = dict()
            for i, s in enumerate(inputs[n]):
                t0 = time()
                summaries = self.summarize(adj, s)
                self.node_weights[n][i] = np.zeros(self.num_nodes[n])
                for j, sum in zip(self.avail_node_set[n][i], summaries):
                    self.node_weights[n][i][j] = counts[sum] ** (-self.sampling_bias)
                self.state_weights[n][i] = np.sum(self.node_weights[n][i])
                t1 = time()
                if self.verbose:
                    p_bar.set_description(
                        "Update weights - " + str(round(t1 - t0, 5)) + "s"
                    )
                    p_bar.update()
            self.graph_weights[n] = np.sum(
                [self.state_weights[n][i] for i in self.state_set[n]]
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
    def __init__(self, name, dynamics, config, verbose=0):
        super(StateBiasedSampler, self).__init__(name, config, verbose)
        self.dynamics_states = list(dynamics.state_label.values())

    def summarize(self, adj, state):
        summaries_arr = np.zeros((state.shape[0], len(self.dynamics_states) + 1))
        summaries_arr[:, 0] = state
        for s in self.dynamics_states:
            summaries_arr[:, int(s) + 1] = np.matmul(adj, state == s)

        summaries = [tuple(s) for s in summaries_arr]
        return summaries
