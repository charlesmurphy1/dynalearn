from .base import Metrics
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from abc import abstractmethod


class LTPMetrics(Metrics):
    def __init__(self, config, verbose=0):
        self.__config = config
        self.aggregator = config.aggregator
        self.max_num_sample = config.max_num_sample
        self.num_points = config.num_points
        super(LTPMetrics, self).__init__(verbose)

    @abstractmethod
    def get_metric(self, adj, inputs, targets):
        raise NotImplementedError("get_metric must be implemented.")

    def aggregate(
        self,
        data=None,
        in_state=None,
        out_state=None,
        for_degree=False,
        err_operation="std",
    ):
        if self.aggregate is None:
            return

        if data is None:
            data = self.data["ltp/train"]

        x, y, err_low, err_high = self.aggregator(
            self.data["summaries"],
            data,
            in_state=in_state,
            out_state=out_state,
            for_degree=for_degree,
            operation="mean",
            err_operation=err_operation,
        )
        return x, y, err_low, err_high

    def compare(self, name1, name2, metrics, in_state=None, out_state=None, func=None):
        ans = np.zeros(self.data["summaries"].shape[0])
        for i, s in enumerate(self.data["summaries"]):
            index = np.where(np.prod(metrics.data["summaries"] == s, axis=-1) == 1)[0]
            if len(index) > 0:
                ans[i] = func(self.data[name1][i], metrics.data[name2][index[0]])
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
            if self.num_points is None:
                n[g] = inputs[g].shape[0]
            elif self.num_points < inputs[g].shape[0]:
                n[g] = self.num_points
            else:
                n[g] = inputs[g].shape[0]

        if self.verbose != 0:
            print("Computing " + self.__class__.__name__)
        if self.verbose == 1:
            num_iter = int(np.sum([inputs[g].shape[0] for g in graphs]))
            if self.num_points < num_iter:
                num_iter = self.num_points
            p_bar = tqdm.tqdm(range(num_iter))

        for g in graphs:
            experiment.model.adj = graphs[g]
            experiment.dynamics_model.adj = graphs[g]
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
                predictions = self.get_metric(experiment, x, y)
                summaries, counter = self.summarize(
                    summaries,
                    counter,
                    predictions,
                    graphs[g],
                    x,
                    y,
                    state_label,
                    train_nodes,
                    val_nodes,
                    test_nodes,
                )

                if self.verbose == 1:
                    p_bar.update()

        if self.verbose == 1:
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
    def __init__(self, config, verbose=0):
        super(TrueLTPMetrics, self).__init__(config, verbose)

    def get_metric(self, experiment, input, target):
        return experiment.dynamics_model.predict(input)


class GNNLTPMetrics(LTPMetrics):
    def __init__(self, config, verbose=0):
        super(GNNLTPMetrics, self).__init__(config, verbose)

    def get_metric(self, experiment, input, target):
        return experiment.model.predict(input)


class MLELTPMetrics(LTPMetrics):
    def __init__(self, config, verbose=0):
        super(MLELTPMetrics, self).__init__(config, verbose)
        if "mle_num_points" in config.__dict__:
            self.num_points = config.mle_num_points

    def get_metric(self, experiment, input, target):
        one_hot_target = np.zeros(
            (target.shape[0], len(experiment.dynamics_model.state_label)), dtype="int"
        )
        one_hot_target[np.arange(target.shape[0]), target.astype("int")] = 1
        return one_hot_target
