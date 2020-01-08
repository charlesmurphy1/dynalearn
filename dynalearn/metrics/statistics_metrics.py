from .base import Metrics
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
from scipy.special import binom
import tqdm


class StatisticsMetrics(Metrics):
    def __init__(self, config, verbose=1):
        super(StatisticsMetrics, self).__init__(verbose)
        self.__config = config
        self.aggregator = config.aggregator
        self.num_points = config.num_points

    def aggregate(
        self, in_state=None, out_state=None, for_degree=False, dataset="train"
    ):
        x, y, _, _ = self.aggregator(
            self.data["summaries"],
            self.data["counts/" + dataset],
            in_state=in_state,
            out_state=out_state,
            for_degree=for_degree,
            operation="sum",
        )
        return x, y

    def summarize(
        self,
        summaries,
        adj,
        input,
        state_label,
        train_nodes,
        val_nodes=[],
        test_nodes=[],
    ):
        infected_deg = np.array(
            [np.matmul(adj, input == state_label[s]) for s in state_label]
        ).T
        for i in range(input.shape[0]):
            s = (input[i], *list(infected_deg[i]))
            if i in train_nodes:
                k = "train"
            elif i in val_nodes:
                k = "val"
            elif i in test_nodes:
                k = "test"
            if s not in summaries:
                summaries[s] = {}
                summaries[s]["train"] = 0
                summaries[s]["val"] = 0
                summaries[s]["test"] = 0
                summaries[s][k] = 1
            else:
                summaries[s][k] += 1
        return summaries

    def compute(self, experiment):
        graphs = experiment.generator.graphs
        inputs = experiment.generator.inputs
        state_label = experiment.dynamics_model.state_label
        train_nodeset = experiment.generator.samplers["train"].avail_node_set

        if "val" in experiment.generator.samplers:
            val_nodeset = experiment.generator.samplers["val"].avail_node_set
        else:
            val_nodeset = None

        if "test" in experiment.generator.samplers:
            test_nodeset = experiment.generator.samplers["test"].avail_node_set
        else:
            test_nodeset = None

        summaries = {}
        n = {}
        for g in graphs:
            if self.num_points < inputs[g].shape[0]:
                n[g] = self.num_points
            else:
                n[g] = inputs[g].shape[0]

        if self.verbose:
            num_iter = int(np.sum([inputs[g].shape[0] for g in graphs]))
            if self.num_points < num_iter:
                num_iter = self.num_points
            p_bar = tqdm.tqdm(range(num_iter), "Computing " + self.__class__.__name__)

        for g in graphs:
            adj = graphs[g]
            for t in range(n[g]):
                x = inputs[g][t]
                train_nodes = train_nodeset[g][t]
                if val_nodeset is not None:
                    val_nodes = val_nodeset[g][t]
                else:
                    val_nodes = []
                if test_nodeset is not None:
                    test_nodes = test_nodeset[g][t]
                else:
                    test_nodes = []

                summaries = self.summarize(
                    summaries, adj, x, state_label, train_nodes, val_nodes, test_nodes
                )
                if self.verbose:
                    p_bar.update()
        if self.verbose:
            p_bar.close()

        self.data["summaries"] = np.array([s for s in summaries])
        self.data["counts/train"] = np.array(
            [summaries[s]["train"] if "train" in summaries[s] else 0 for s in summaries]
        )
        self.data["counts/val"] = np.array(
            [summaries[s]["val"] if "val" in summaries[s] else 0 for s in summaries]
        )
        self.data["counts/test"] = np.array(
            [summaries[s]["test"] if "test" in summaries[s] else 0 for s in summaries]
        )

    def jensenshannon(self, dataset1, dataset2):
        prob1 = self.data["counts/" + dataset1] / np.sum(
            self.data["counts/" + dataset1]
        )
        prob2 = self.data["counts/" + dataset2] / np.sum(
            self.data["counts/" + dataset2]
        )
        return jensenshannon(prob1, prob2)

    def overlap(self, dataset1, dataset2):
        prob1 = self.data["counts/" + dataset1] / np.sum(
            self.data["counts/" + dataset1]
        )
        prob2 = self.data["counts/" + dataset2] / np.sum(
            self.data["counts/" + dataset2]
        )
        return 1 - abs(prob1 - prob2) / 2

    def entropy(self, dataset, normalize=True):
        prob = self.data["counts/" + dataset] / np.sum(self.data["counts/" + dataset])
        entropy = -np.sum(prob[prob > 0] * np.log(prob[prob > 0]))

        if normalize:
            x = prob > 0
            prob_uni = x / np.sum(x)
            entropy_uni = -np.sum(
                prob_uni[prob_uni > 0] * np.log(prob_uni[prob_uni > 0])
            )
            entropy /= entropy_uni
        return entropy

    def max_entropy(self, dataset):
        degrees = np.unique(np.sort(np.sum(self.data["summaries"][:, 1:], axis=-1)))
        degrees = degrees[degrees > 0]
        num_states = self.data["summaries"][0, 1:].shape[0]
        ans = 0
        for k in degrees:
            ans += num_states * binom(num_states + k - 1, k)
        max_entropy = np.log(ans)
        return max_entropy

    def effective_samplesize(self, dataset):
        w = self.data["counts/" + dataset]
        return np.sum(w) ** 2 / np.sum(w ** 2)
