from .base_metrics import Metrics
import matplotlib.pyplot as plt
import numpy as np
import tqdm


class CountMetrics(Metrics):
    def __init__(self, aggregator=None, num_points=1000, verbose=1):
        super(CountMetrics, self).__init__(verbose)
        self.aggregator = aggregator
        self.num_points = num_points

    def display(self, in_state, dataset, for_degree=False, ax=None, **bar_kwargs):
        if ax is None:
            ax = plt.gca()
        if "counts/" + dataset not in self.data or self.aggregator is None:
            return ax
        if not for_degree:
            x, y, err = self.aggregator(
                in_state,
                self.data["summaries"],
                self.data["counts/" + dataset],
                operation="sum",
            )
        else:
            x = np.unique(np.sort(np.sum(self.data["summaries"][:, 1:], axis=-1)))
            y = np.zeros(x.shape)
            for i, xx in enumerate(x):
                index = (np.sum(self.data["summaries"][:, 1:], axis=-1) == xx) * (
                    self.data["summaries"][:, 0] == in_state
                )
                y[i] = np.sum(self.data["counts/" + dataset][index])
        bar_width = abs(x[1] - x[0]) / (self.data["summaries"].shape[1])
        ax.bar(x + in_state * bar_width, y / np.sum(y), bar_width, **bar_kwargs)
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
            num_iter = np.sum([inputs[g].shape[0] for g in graphs])
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

    def overlap(self, dataset1, dataset2):
        prob1 = self.data["counts/" + dataset1] / np.sum(
            self.data["counts/" + dataset1]
        )
        prob2 = self.data["counts/" + dataset2] / np.sum(
            self.data["counts/" + dataset2]
        )
        return 1 - np.sum(abs(prob1 - prob2)) / 2
