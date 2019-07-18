from .base_metrics import Metrics
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import iqr, gaussian_kde
import tqdm


class LossMetrics(Metrics):
    def __init__(self, num_points=1000, max_num_sample=10000, verbose=1):
        super(LossMetrics, self).__init__(verbose)
        self.num_points = num_points
        self.max_num_sample = max_num_sample
        self.is_full = False
        self.datasets = []

    def loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-8, None)
        return -np.sum(y_true * np.log(y_pred), axis=-1)

    def display(
        self,
        loss_name,
        dataset,
        width=None,
        ax=None,
        color=None,
        hist_alpha=0.3,
        kde_linewidth=2,
        kde_linestyle="-",
        rug_pos=0,
        hist=True,
        kde=True,
        rug=False,
    ):
        if ax is None:
            ax = plt.gca()

        samples = self.data[f"{dataset}/{loss_name}"]
        if width is None:
            if iqr(samples) > 0:
                width = 2 * iqr(samples) / samples.shape[0] ** (1.0 / 3)
            else:
                width = 1e-3
        if hist:
            bins = np.arange(np.min(samples), np.max(samples), width)
            histo, bins = np.histogram(samples, bins=bins, density=True)
            x_hist = 0.5 * (bins[1:] + bins[:-1])
            ax.bar(x_hist, histo, width=width, color=color, alpha=0.3)

        if kde:
            kernel = gaussian_kde(samples)
            x_kde = np.arange(np.min(samples), np.max(samples), width)
            ax.plot(
                x_kde,
                kernel.pdf(x_kde),
                color=color,
                marker="None",
                linestyle=kde_linestyle,
                linewidth=kde_linewidth,
            )

        if rug:
            y_min, y_max = ax.get_ylim()
            rug_pos *= y_max - y_min
            ax.plot(samples, [rug_pos] * len(samples), "|", color=color)

        return ax

    def compute(self, experiment):
        if self.is_full:
            return

        model = experiment.model
        graphs = experiment.generator.graphs
        inputs = experiment.generator.inputs
        targets = experiment.generator.targets
        gt_targets = experiment.generator.gt_targets
        node_weights = experiment.generator.sampler.node_weights

        num_states = len(experiment.dynamics_model.state_label)

        nodeset = {}
        nodeset["train"] = experiment.generator.sampler.avail_node_set
        self.datasets.append("train")

        if experiment.val_generator is not None:
            nodeset["val"] = experiment.val_generator.sampler.avail_node_set
            self.datasets.append("val")

        if experiment.test_generator is not None:
            nodeset["test"] = experiment.test_generator.sampler.avail_node_set
            self.datasets.append("test")

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

        approx_loss_dict = {d: np.zeros(self.max_num_sample) for d in self.datasets}
        exact_loss_dict = {d: np.zeros(self.max_num_sample) for d in self.datasets}
        diff_loss_dict = {d: np.zeros(self.max_num_sample) for d in self.datasets}
        counter = {d: 0 for d in self.datasets}

        for g in graphs:
            if self.is_full:
                break
            adj = graphs[g]
            for t in range(n[g]):
                if self.is_full:
                    break
                x = inputs[g][t]
                approx_y_true = self.to_one_hot(targets[g][t], num_states)
                exact_y_true = gt_targets[g][t]
                y_pred = model.predict(x, adj)
                approx_loss = self.loss(approx_y_true, y_pred)
                exact_loss = self.loss(exact_y_true, y_pred)
                for d in self.datasets:
                    mask = np.zeros(x.shape[0])
                    mask[nodeset[d][g][t]] = 1
                    mask /= np.sum(mask)
                    l1 = np.sum(mask * approx_loss)
                    l2 = np.sum(mask * exact_loss)
                    l3 = abs(l1 - l2)
                    approx_loss_dict[d][counter[d]] = l1
                    exact_loss_dict[d][counter[d]] = l2
                    diff_loss_dict[d][counter[d]] = l3
                    counter[d] += 1
                    if counter[d] == self.max_num_sample:
                        self.is_full = True

                if self.verbose:
                    p_bar.update()

        if self.verbose:
            p_bar.close()

        for d in self.datasets:
            self.data[d + "/approx_loss"] = approx_loss_dict[d][: counter[d]]
            self.data[d + "/exact_loss"] = exact_loss_dict[d][: counter[d]]
            self.data[d + "/diff_loss"] = diff_loss_dict[d][: counter[d]]

    def to_one_hot(self, arr, num_states):
        ans = np.zeros((arr.shape[0], num_states), dtype="int")
        ans[np.arange(arr.shape[0]), arr.astype("int")] = 1
        return ans
