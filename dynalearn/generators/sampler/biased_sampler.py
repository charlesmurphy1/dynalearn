from .random_sampler import RandomSampler
from .sampling_method import DiscreteSampling
import numpy as np
from time import time
import tqdm


class BiasedSampler(RandomSampler):
    def __init__(self, sampling_bias=0, batch_size=None, replace=False, verbose=1):
        super(BiasedSampler, self).__init__(batch_size, replace, verbose)
        self.params["sampling_bias"] = sampling_bias

    def update_weights(self, graphs, inputs):
        if self.verbose:
            nn = np.sum([inputs[n].shape[0] for n in graphs])
            p_bar = tqdm.tqdm(range(nn))

        # Update counts
        counts = dict()
        for n in graphs:
            adj = graphs[n]
            for state in inputs[n]:
                t0 = time()
                summaries = self.summarize(adj, state)
                # print(summaries)
                for s in summaries:
                    # s = tuple(s)
                    if s in counts:
                        counts[s] += 1
                    else:
                        counts[s] = 1
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

        # Update graph and node weights
        self.node_weights = dict()
        self.state_weights = dict()
        self.graph_weights = dict()
        for n in graphs:
            adj = graphs[n]
            self.node_weights[n] = np.zeros(inputs[n].shape)
            for i, state in enumerate(inputs[n]):
                t0 = time()
                summary = self.summarize(adj, state)
                for j, s in enumerate(summary):
                    self.node_weights[n][i, j] = counts[s] ** (
                        -self.params["sampling_bias"]
                    )
                t1 = time()
                if self.verbose:
                    p_bar.set_description(
                        "Update weights - " + str(round(t1 - t0, 5)) + "s"
                    )
                    p_bar.update()
            self.state_weights[n] = {
                i: np.sum(self.node_weights[n][int(i), :])
                for i in self.avail_state_set[n]
            }
            self.graph_weights[n] = np.sum(
                [self.node_weights[n][int(i), :] for i in self.avail_state_set[n]]
            )
        if self.verbose:
            p_bar.close()

    def summarize(self, adj, state):
        raise NotImplementedError()


class DegreeBiasedSampler(BiasedSampler):
    def __init__(self, sampling_bias=0, batch_size=None, replace=False, verbose=1):
        super(DegreeBiasedSampler, self).__init__(
            sampling_bias, batch_size, replace, verbose
        )

    def summarize(self, adj, state):
        return np.sum(adj, axis=0)


class StateBiasedSampler(BiasedSampler):
    def __init__(
        self, dynamics, sampling_bias=0, batch_size=None, replace=False, verbose=1
    ):
        super(StateBiasedSampler, self).__init__(
            sampling_bias, batch_size, replace, verbose
        )
        self.dynamics_states = list(dynamics.state_label.values())

    def summarize(self, adj, state):
        summaries_arr = np.zeros((state.shape[0], len(self.dynamics_states) + 1))
        summaries_arr[:, 0] = state
        for s in self.dynamics_states:
            summaries_arr[:, int(s) + 1] = np.matmul(adj, state == s)

        summaries = [tuple(s) for s in summaries_arr]
        return summaries
