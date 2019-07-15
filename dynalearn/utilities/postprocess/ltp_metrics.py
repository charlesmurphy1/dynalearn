from .base_metrics import Metrics
import matplotlib.pyplot as plt
import numpy as np
import tqdm


class LTPMetrics(Metrics):
    def __init__(self, num_points=1000, verbose=1):
        super(LTPMetrics, self).__init__(verbose)
        self.num_points = num_points

    def get_metric(self, adj, inputs, targets):
        raise NotImplementedError()

    def display(
        self,
        in_state,
        out_state,
        neighbor_state,
        dataset,
        ax=None,
        fill=None,
        **plot_kwargs
    ):
        if "ltp_mu1/" + dataset not in self.data:
            return
        if ax is None:
            ax = plt.gca()
        x = np.unique(np.sort(self.data["summaries"][:, neighbor_state + 1]))
        y = np.zeros(x.shape)
        err = np.zeros(x.shape)
        for i, xx in enumerate(x):
            index = (self.data["summaries"][:, neighbor_state + 1] == xx) * (
                self.data["summaries"][:, 0] == in_state
            )
            mu1 = self.data["ltp_mu1/" + dataset][index, out_state]
            mu2 = self.data["ltp_mu2/" + dataset][index, out_state]
            counts = self.data["counts/" + dataset][index]
            y[i] = np.mean(mu1[~np.isnan(mu1)])
            err[i] = np.sqrt(
                (np.mean(mu2[~np.isnan(mu2)]) - np.mean(mu1[~np.isnan(mu1)]) ** 2)
                / np.sum(counts)
            )
        if fill is not None:
            ax.fill_between(x, y - err, y + err, color=fill, alpha=0.3)
        ax.plot(x, y, **plot_kwargs)
        return ax

    def summarize(
        self,
        summaries,
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
            pred = predictions[i].reshape(1, -1)

            if i in train_nodes:
                k = "train"
            elif i in val_nodes:
                k = "val"
            elif i in test_nodes:
                k = "test"
            if s in summaries:
                if k in summaries[s]:
                    summaries[s][k] = np.append(summaries[s][k], pred, axis=0)
                else:
                    summaries[s][k] = pred
            else:
                summaries[s] = {}
                summaries[s][k] = pred
        return summaries

    def compute(self, experiment):

        graphs = experiment.generator.graphs
        inputs = experiment.generator.inputs
        targets = experiment.generator.targets
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
                summaries = self.summarize(
                    summaries,
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
            self.data["ltp_mu1/" + k] = np.array(
                [
                    [*np.mean(summaries[tuple(s)][k], axis=0)]
                    if k in summaries[tuple(s)]
                    else [np.nan] * d
                    for s in self.data["summaries"]
                ]
            )
            self.data["ltp_mu2/" + k] = np.array(
                [
                    [*np.mean(summaries[tuple(s)][k] ** 2, axis=0)]
                    if k in summaries[tuple(s)]
                    else [np.nan] * d
                    for s in self.data["summaries"]
                ]
            )
            self.data["counts/" + k] = np.array(
                [
                    summaries[tuple(s)][k].shape[0] if k in summaries[tuple(s)] else 0
                    for s in self.data["summaries"]
                ]
            )


class DynamicsLTPMetrics(LTPMetrics):
    def __init__(self, num_points=1000, verbose=1):
        super(DynamicsLTPMetrics, self).__init__(num_points, verbose)

    def get_metric(self, experiment, adj, input, target):
        return experiment.dynamics_model.predict(input, adj)


class ModelLTPMetrics(LTPMetrics):
    def __init__(self, num_points=1000, verbose=1):
        super(ModelLTPMetrics, self).__init__(num_points, verbose)

    def get_metric(self, experiment, adj, input, target):
        return experiment.model.predict(input, adj)


class EstimatorLTPMetrics(LTPMetrics):
    def __init__(self, num_points=1000, verbose=1):
        super(EstimatorLTPMetrics, self).__init__(num_points, verbose)

    def get_metric(self, experiment, adj, input, target):
        one_hot_target = np.zeros(
            (target.shape[0], len(experiment.dynamics_model.state_label)), dtype="int"
        )
        one_hot_target[np.arange(target.shape[0]), target.astype("int")] = 1
        return one_hot_target
