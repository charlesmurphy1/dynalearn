from .base_metrics import Metrics
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
import tqdm


class CountMetrics(Metrics):
    def __init__(self, aggregator=None, num_points=1000, verbose=1):
        super(CountMetrics, self).__init__(verbose)
        self.aggregator = aggregator
        self.num_points = num_points

    def aggregate(self, in_state=None, for_degree=False, dataset="train"):
        if not for_degree:
            x, y, _ = self.aggregator(
                self.data["summaries"],
                self.data["counts/" + dataset],
                in_state=in_state,
                operation="sum",
            )
            _, z, _ = self.aggregator(
                None,
                self.data["summaries"],
                self.data["counts/" + dataset],
                operation="sum",
            )
            y /= np.sum(z)
        else:
            x = np.unique(np.sort(np.sum(self.data["summaries"][:, 1:], axis=-1)))
            x = x[x > 0]
            y = np.zeros(x.shape)
            for i, xx in enumerate(x):
                if in_state is None:
                    index = np.sum(self.data["summaries"][:, 1:], axis=-1) == xx
                else:
                    index = (np.sum(self.data["summaries"][:, 1:], axis=-1) == xx) * (
                        self.data["summaries"][:, 0] == in_state
                    )
                y[i] = np.sum(self.data["counts/" + dataset][index])
            y /= np.sum(y)

        return x, y

    def display(
        self,
        in_state,
        dataset,
        for_degree=False,
        ax=None,
        line=False,
        color="k",
        **kwargs
    ):
        if ax is None:
            ax = plt.gca()
        if "counts/" + dataset not in self.data or self.aggregator is None:
            return ax
        offset = in_state
        bar_alpha = 1
        if offset is None:
            offset = 0
        if not for_degree:
            x, y, err = self.aggregator(
                self.data["summaries"],
                self.data["counts/" + dataset],
                in_state=in_state,
                operation="sum",
            )
            _x, _y, _err = self.aggregator(
                None,
                self.data["summaries"],
                self.data["counts/" + dataset],
                operation="sum",
            )
            y /= np.sum(_y)
            bar_width = np.nanmean(abs(x[1:] - np.roll(x, 1)[1:]))
        else:
            x = np.unique(np.sort(np.sum(self.data["summaries"][:, 1:], axis=-1)))
            x = x[x > 0]
            y = np.zeros(x.shape)
            for i, xx in enumerate(x):
                if in_state is None:
                    index = np.sum(self.data["summaries"][:, 1:], axis=-1) == xx
                else:
                    index = (np.sum(self.data["summaries"][:, 1:], axis=-1) == xx) * (
                        self.data["summaries"][:, 0] == in_state
                    )
                y[i] = np.sum(self.data["counts/" + dataset][index])
            y /= np.sum(y)
            bar_width = np.nanmin(abs(x[1:] - np.roll(x, 1)[1:]))
        if in_state is not None:
            bar_width /= self.data["summaries"].shape[1]

        if line:
            ax.plot(x, y, color=color, **kwargs)
            bar_alpha = 0.3
        ax.bar(x + offset * bar_width, y, bar_width, color=color, alpha=bar_alpha)
        return ax

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
        train_nodeset = experiment.generator.sampler.avail_node_set

        if experiment.val_generator is not None:
            val_nodeset = experiment.val_generator.sampler.avail_node_set
        else:
            val_nodeset = None
        if experiment.test_generator is not None:
            test_nodeset = experiment.test_generator.sampler.avail_node_set
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
