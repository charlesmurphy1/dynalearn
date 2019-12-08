from .base import Metrics
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import iqr, gaussian_kde
import tqdm


class LossMetrics(Metrics):
    def __init__(self, config, verbose=1):
        super(LossMetrics, self).__init__(verbose)
        self.__config = config
        self.num_points = config.num_points
        self.max_num_sample = config.max_num_sample
        self.datasets = []

    def loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-8, 1.0 - 1e-8)
        return -np.sum(y_true * np.log(y_pred), axis=-1)

    def compute(self, experiment):
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
            num_iter = int(np.sum([inputs[g].shape[0] for g in graphs]))
            if self.num_points < num_iter:
                num_iter = self.num_points
            p_bar = tqdm.tqdm(range(num_iter), "Computing " + self.__class__.__name__)

        approx_loss_dict = {d: np.zeros(self.max_num_sample) for d in self.datasets}
        exact_loss_dict = {d: np.zeros(self.max_num_sample) for d in self.datasets}
        diff_loss_dict = {d: np.zeros(self.max_num_sample) for d in self.datasets}
        counter = {d: 0 for d in self.datasets}
        is_full = False
        for g in graphs:
            if is_full:
                break
            adj = graphs[g]
            for t in range(n[g]):
                if is_full:
                    break
                x = inputs[g][t]
                approx_y_true = self.to_one_hot(targets[g][t], num_states)
                exact_y_true = gt_targets[g][t]
                y_pred = model.predict(x, adj)
                p = node_weights[g][t] / np.sum(node_weights[g][t])

                approx_loss = self.loss(approx_y_true, y_pred)
                exact_loss = self.loss(exact_y_true, y_pred)
                for d in self.datasets:
                    mask = np.zeros(x.shape[0])
                    mask[nodeset[d][g][t]] = p[nodeset[d][g][t]] * len(nodeset[d][g][t])
                    l1 = np.mean(mask * approx_loss)
                    l2 = np.mean(mask * exact_loss)
                    l3 = abs(l1 - l2)
                    approx_loss_dict[d][counter[d]] = l1
                    exact_loss_dict[d][counter[d]] = l2
                    diff_loss_dict[d][counter[d]] = l3
                    counter[d] += 1
                    if counter[d] == self.max_num_sample:
                        is_full = True

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
