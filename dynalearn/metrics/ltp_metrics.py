from .base_metrics import Metrics
import matplotlib.pyplot as plt
import numpy as np
import tqdm


class LTPMetrics(Metrics):
    def __init__(
        self, aggregator=None, num_points=None, max_num_sample=1000, verbose=1
    ):
        self.aggregator = aggregator
        self.max_num_sample = max_num_sample
        if num_points is None:
            self.num_points = np.inf
        else:
            self.num_points = num_points
        super(LTPMetrics, self).__init__(verbose)

    def get_metric(self, adj, inputs, targets):
        raise NotImplementedError()

    def aggregate(self, data=None, in_state=None, out_state=None, for_degree=False):
        if self.aggregate is None:
            return

        if data is None:
            data = self.data["ltp/train"]

        x, y, err = self.aggregator(
            self.data["summaries"],
            data,
            in_state=in_state,
            out_state=out_state,
            for_degree=for_degree,
            operation="mean",
        )
        return x, y, err

    def compare(self, name1, name2, metrics, in_state=None, out_state=None, func=None):
        ans = np.zeros(self.data["summaries"].shape[0])
        for i, s in enumerate(self.data["summaries"]):
            index = np.where(np.prod(metrics.data["summaries"] == s, axis=-1) == 1)[0]
            if len(index) > 0:
                ans[i] = func(self.data[name1][i], metrics.data[name2][index])
            else:
                ans[i] = np.nan
        return ans

    def summarize(
        self,
        summaries,
        counter,
        predictions,
        adj,
        input,
        target,
        state_label,
        train_nodes,
        val_nodes=[],
        test_nodes=[],
    ):
        state_deg = np.array(
            [np.matmul(adj, input == state_label[s]) for s in state_label]
        ).T
        for i in range(input.shape[0]):
            s = (input[i], *list(state_deg[i]))
            pred = predictions[i]

            if i in train_nodes:
                k = "train"
            elif i in val_nodes:
                k = "val"
            elif i in test_nodes:
                k = "test"
            if s in summaries:
                if k in summaries[s]:
                    if counter[s][k] == self.max_num_sample:
                        continue
                    summaries[s][k][counter[s][k], :] = pred
                    counter[s][k] += 1
                else:
                    summaries[s][k] = np.zeros((self.max_num_sample, pred.shape[0]))
                    summaries[s][k][0] = pred
                    counter[s][k] = 1
            else:
                summaries[s] = {}
                counter[s] = {}
                summaries[s][k] = np.zeros((self.max_num_sample, pred.shape[0]))
                summaries[s][k][0] = pred
                counter[s][k] = 1

        return summaries, counter

    def compute(self, experiment):

        graphs = experiment.generator.graphs
        inputs = experiment.generator.inputs
        targets = experiment.generator.targets
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
        counter = {}
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
                y = targets[g][t]
                train_nodes = train_nodeset[g][t]
                if val_nodeset is not None:
                    val_nodes = val_nodeset[g][t]
                else:
                    val_nodes = []
                if test_nodeset is not None:
                    test_nodes = test_nodeset[g][t]
                else:
                    test_nodes = []
                predictions = self.get_metric(experiment, adj, x, y)
                summaries, counter = self.summarize(
                    summaries,
                    counter,
                    predictions,
                    adj,
                    x,
                    y,
                    state_label,
                    train_nodes,
                    val_nodes,
                    test_nodes,
                )

                if self.verbose:
                    p_bar.update()

        if self.verbose:
            p_bar.close()

        d = len(state_label)

        self.data["summaries"] = np.array([[*s] for s in summaries])
        for k in ["train", "val", "test"]:
            self.data["ltp/" + k] = np.array(
                [
                    np.mean(
                        summaries[tuple(s)][k][: counter[tuple(s)][k], :], axis=0
                    ).squeeze()
                    if k in summaries[tuple(s)]
                    else [np.nan] * d
                    for s in self.data["summaries"]
                ]
            )
            self.data["counts/" + k] = np.array(
                [
                    counter[tuple(s)][k] if k in summaries[tuple(s)] else 0
                    for s in self.data["summaries"]
                ]
            )

    def entropy(self, dataset):
        ltp = self.data["ltp/" + dataset]
        counts = self.data["counts/" + dataset]
        ent = np.zeros(ltp.shape[0])
        for i, l in enumerate(ltp):
            c = counts[i]
            ent[i] = -np.sum(l[l > 0] * np.log(l[l > 0]))
        return np.nansum(ent * counts) / np.sum(counts)


class TrueLTPMetrics(LTPMetrics):
    def __init__(
        self, aggregator=None, num_points=1000, max_num_sample=1000, verbose=1
    ):
        super(TrueLTPMetrics, self).__init__(
            aggregator, num_points, max_num_sample, verbose
        )

    def get_metric(self, experiment, adj, input, target):
        return experiment.dynamics_model.predict(input, adj)


class GNNLTPMetrics(LTPMetrics):
    def __init__(
        self, aggregator=None, num_points=1000, max_num_sample=1000, verbose=1
    ):
        super(GNNLTPMetrics, self).__init__(
            aggregator, num_points, max_num_sample, verbose
        )

    def get_metric(self, experiment, adj, input, target):
        return experiment.model.predict(input, adj)


class MLELTPMetrics(LTPMetrics):
    def __init__(
        self, aggregator=None, num_points=1000, max_num_sample=1000, verbose=1
    ):
        super(MLELTPMetrics, self).__init__(
            aggregator, num_points, max_num_sample, verbose
        )

    def get_metric(self, experiment, adj, input, target):
        one_hot_target = np.zeros(
            (target.shape[0], len(experiment.dynamics_model.state_label)), dtype="int"
        )
        one_hot_target[np.arange(target.shape[0]), target.astype("int")] = 1
        return one_hot_target
