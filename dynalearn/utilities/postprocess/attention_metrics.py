from .base_metrics import Metrics
from dynalearn.models.layers import GraphAttention
from itertools import product
import numpy as np
from scipy.stats import iqr, gaussian_kde
import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

# from seaborn import distplot


class AttentionMetrics(Metrics):
    def __init__(self, num_points=100, max_num_sample=10000, verbose=1):
        super(AttentionMetrics, self).__init__(verbose)
        self.num_points = num_points
        self.max_num_sample = max_num_sample
        self.num_layers = 0

    def _get_all_attn_layers(self, experiment):

        attn_layers = []
        num_attn = 0
        for layer in experiment.model.model.layers:
            if type(layer) == GraphAttention:
                num_attn += 1
                attn_layers.append(experiment.model.get_attn_layer(num_attn))

        return attn_layers

    def display(self, layer, state_label, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        d = len(state_label)
        mean_attn = np.zeros((d, d))
        for i in range(len(state_label)):
            for j in range(len(state_label)):
                mean_attn[j, i] = np.mean(self.data[f"layer{layer}/{i}_{j}"])
        ax.imshow(mean_attn, extent=[0, d, 0, d], origin="lower", **kwargs)
        return ax

    def display_dist(
        self,
        layer,
        node_state,
        neighbor_state,
        ax=None,
        color=None,
        width=None,
        kde=False,
        rug=False,
        hist=False,
        box=True,
    ):
        if ax is None:
            ax = plt.gca()
        samples = self.data[f"layer{layer}/{node_state}_{neighbor_state}"]
        if width is None:
            if iqr(samples) > 0:
                width = 2 * iqr(samples) / samples.shape[0] ** (1.0 / 3)
            else:
                width = 1e-2
        if hist:
            bins = np.arange(np.min(samples), np.max(samples), width)
            histo, bins = np.histogram(samples, bins=bins, density=True)
            x_hist = 0.5 * (bins[1:] + bins[:-1])
            ax.bar(x_hist, histo, width=width, color=color)

        if rug:
            ax.plot(samples, [0.01] * len(samples), "|", color="k")

        if kde:
            kernel = gaussian_kde(samples)
            x_kde = np.arange(-width, 1 + width, width)
            ax.plot(x_kde, kernel.pdf(x_kde), color=color)

        if box:
            ax.boxplot(samples, vert=False)

        return ax

    def summarize(self, summaries, prediction, state_label, adj, inputs):
        infected_deg = np.array(
            [np.matmul(adj, inputs == state_label[s]) for s in state_label]
        ).T

        for i, j in zip(*np.nonzero(adj)):
            s = (inputs[i], inputs[j])
            if len(summaries[s]) < self.max_num_sample:
                summaries[s] = np.append(summaries[s], prediction[i, j])
            else:
                continue
        return summaries

    def compute(self, experiment):
        graphs = experiment.generator.graphs
        inputs = experiment.generator.inputs
        state_label = experiment.dynamics_model.state_label

        n = {}
        for g in graphs:
            if self.num_points < inputs[g].shape[0]:
                n[g] = self.num_points
            else:
                n[g] = inputs[g].shape[0]
        state_label = experiment.dynamics_model.state_label
        attention_layers = self._get_all_attn_layers(experiment)

        if self.verbose:
            num_iter = int(len(attention_layers) * np.sum([n[g] for g in graphs]))
            p_bar = tqdm.tqdm(range(num_iter), "Computing " + self.__class__.__name__)

        for l, layer in enumerate(attention_layers):
            summaries = {
                (i, j): np.array([])
                for i, j in product(state_label.values(), state_label.values())
            }
            for g in graphs:
                adj = graphs[g]
                for t in range(n[g]):
                    if self.is_summaries_full(summaries):
                        break
                    x = inputs[g][t]
                    predictions = layer.predict([x, adj], steps=1)
                    summaries = self.summarize(
                        summaries, predictions, state_label, adj, x
                    )
                    if self.verbose:
                        p_bar.update()
            for s in summaries:
                in_s = int(s[0])
                out_s = int(s[1])
                self.data[f"layer{l}/{in_s}_{out_s}"] = summaries[s]

        if self.verbose:
            p_bar.close()
        self.num_layers = len(attention_layers)

    def is_summaries_full(self, summaries):
        for s in summaries:
            if not len(summaries[s]) > self.max_num_sample:
                return False
        return True