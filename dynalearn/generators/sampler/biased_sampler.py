from .random_sampler import RandomSampler
from .sampling_method import DiscreteSampling
import numpy as np
from time import time
import tqdm


class DegreeSampler(RandomSampler):
    def __init__(
        self,
        batchsize=None,
        sampling_bias=0,
        sample_from_weight=True,
        verbose=1,
        replace=0,
    ):
        super(DegreeSampler, self).__init__(
            batchsize, sample_from_weight, verbose, replace
        )
        self.params["sampling_bias"] = sampling_bias

        self.counts = dict()
        self.node_weights = dict()
        self.graph_sampler = None
        self.with_weights = True

    def get_graph(self):
        return self.graph_sampler.sample()

    def get_nodes(self, g_index, s_index):
        w = np.zeros(self.num_nodes[g_index])
        if self.params["batchsize"] is None or self.params["batchsize"] > len(
            self.avail_nodeset[g_index]
        ):
            w[self.avail_nodeset[g_index]] = 1
            return w
        else:
            w[self.avail_nodeset[g_index]] = self.node_weights[g_index][
                self.avail_nodeset[g_index]
            ]
            p = self.params["batchsize"] * w / np.sum(w)
            if not self.params["sample_from_weight"]:
                return p
            r = np.random.rand(self.num_nodes[g_index])
            mask = (r < p).astype("float")
            return mask

    def sample_nodes(self, g_index, num_nodes):
        weights = self.node_weights[g_index]
        p = weights[self.avail_nodeset[g_index]] / np.sum(
            weights[self.avail_nodeset[g_index]]
        )
        return np.random.choice(
            self.avail_nodeset[g_index], num_nodes, replace=False, p=p
        )

    def update_weights(self, graphs, inputs, targets):

        # Update counts
        for i in graphs:
            degree_sequence = np.sum(graphs[i], axis=0).astype("int")
            for d in degree_sequence:
                if d in self.counts:
                    self.counts[d] += 1
                else:
                    self.counts[d] = 1

        # Update graph and node weights
        graph_weights = dict()
        for i in graphs:
            degree_sequence = np.sum(graphs[i], axis=0).astype("int")
            num_nodes = graphs[i].shape[0]
            self.node_weights[i] = np.zeros(num_nodes)
            for j, k in enumerate(degree_sequence):
                self.node_weights[i][j] = self.counts[k] ** (
                    -self.params["sampling_bias"]
                )
            graph_weights[i] = np.sum(self.node_weights[i])
        self.graph_sampler = DiscreteSampling(graph_weights)


class DiscreteStateSampler(RandomSampler):
    def __init__(
        self,
        dynamics,
        batchsize=None,
        sampling_bias=0,
        sample_from_weight=True,
        verbose=1,
        replace=0,
    ):
        super(DiscreteStateSampler, self).__init__(
            batchsize, sample_from_weight, verbose, replace
        )
        self.params["states"] = list(dynamics.state_label.values())
        self.params["sampling_bias"] = sampling_bias

        self.counts = dict()
        self.node_weights = dict()
        self.state_sampler = dict()
        self.graph_sampler = None
        self.with_weights = True

    def get_graph(self):
        return self.graph_sampler.sample()

    def get_state(self, g_index):
        return self.state_sampler[g_index].sample()

    def get_nodes(self, g_index, s_index):
        w = np.zeros(self.num_nodes[g_index])
        if self.params["batchsize"] is None or self.params["batchsize"] > len(
            self.avail_nodeset[g_index]
        ):
            w[self.avail_nodeset[g_index]] = 1
            return w
        else:
            w[self.avail_nodeset[g_index]] = self.node_weights[g_index][
                s_index, self.avail_nodeset[g_index]
            ]
            p = self.params["batchsize"] * w / np.sum(w)
            if not self.params["sample_from_weight"]:
                return p
            r = np.random.rand(self.num_nodes[g_index])
            mask = (r < p).astype("float")
            return mask

    def sample_nodes(self, g_index, num_nodes):
        weights = np.sum(self.node_weights[g_index], axis=0)
        p = weights[self.avail_nodeset[g_index]] / np.sum(
            weights[self.avail_nodeset[g_index]]
        )
        return np.random.choice(
            self.avail_nodeset[g_index], num_nodes, replace=False, p=p
        )

    def update_weights(self, graphs, inputs, targets):
        if self.verbose:
            nn = np.sum([inputs[n].shape[0] for n in graphs])
            p_bar = tqdm.tqdm(range(nn))

        # Update counts
        for n in graphs:
            adj = graphs[n]
            for state in inputs[n]:
                t0 = time()
                summaries = self.summarize_state(adj, state)
                for s in summaries:
                    s = tuple(s)
                    if s in self.counts:
                        self.counts[s] += 1
                    else:
                        self.counts[s] = 1
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
        graph_weights = dict()
        state_weights = dict()
        for n in graphs:
            adj = graphs[n]
            self.node_weights[n] = np.zeros(inputs[n].shape)
            for i in range(inputs[n].shape[0]):
                t0 = time()
                summary = self.summarize_state(adj, state)
                for j, s in enumerate(summary):
                    s = tuple(s)
                    self.node_weights[n][i, j] = self.counts[s] ** (
                        -self.params["sampling_bias"]
                    )
                t1 = time()
                if self.verbose:
                    p_bar.set_description(
                        "Update weights - " + str(round(t1 - t0, 5)) + "s"
                    )
                    p_bar.update()
            state_weights[n] = {
                i: np.sum(self.node_weights[n][i, :]) for i in range(inputs[n].shape[0])
            }
            graph_weights[n] = np.sum(self.node_weights[n])
        if self.verbose:
            p_bar.close()

        # Define new state and graph samplers
        self.state_sampler = dict()
        for n in graphs:
            self.state_sampler[n] = DiscreteSampling(state_weights[n])
        self.graph_sampler = DiscreteSampling(graph_weights)

    def summarize_state(self, adj, state):
        summaries = np.zeros((state.shape[0], len(self.params["states"]) + 1))
        summaries[:, 0] = state
        for s in self.params["states"]:
            summaries[:, int(s) + 1] = np.matmul(adj, state == s)

        return summaries
