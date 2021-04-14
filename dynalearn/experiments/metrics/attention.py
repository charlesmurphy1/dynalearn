import networkx as nx
import numpy as np
from abc import abstractmethod
from functools import partial
from sklearn.feature_selection import mutual_info_regression
from dynalearn.experiments.metrics import Metrics
from dynalearn.utilities import Verbose
from dynalearn.nn.models import DynamicsGATConv
from dynalearn.networks import MultiplexNetwork
from dynalearn.experiments.metrics._utils.mutual_info import mutual_info


class AttentionMetrics(Metrics):
    def __init__(self, config):
        Metrics.__init__(self, config)
        p = config.__dict__.copy()
        self.max_num_points = p.pop("max_num_points", np.inf)
        self.indices = {}

    def initialize(self, experiment):
        self.model = experiment.model
        self.datasets = {
            "all": experiment.dataset,
            "train": experiment.dataset,
            "val": experiment.val_dataset,
            "test": experiment.test_dataset,
        }
        for n, d in self.datasets.items():
            if d is None:
                continue
            elif n == "all":
                self.indices[n] = self._get_indices_(d, doall=True)
            else:
                self.indices[n] = self._get_indices_(d, doall=False)

        if isinstance(self.model.nn.gnn_layer, DynamicsGATConv):
            if self.model.config.is_multiplex:
                self.layers = layer = self.model.config.network_layers
            else:
                layers = [None]
            for l in layers:
                name = "attcoeffs"
                if l is not None:
                    name += f"-{l}"
                for n, d in self.datasets.items():
                    if d is None:
                        continue
                    self.names.append(f"{n}-" + name)
                    self.get_data[f"{n}-" + name] = partial(
                        self._compute_feature_, self._get_attcoeffs_, key=n, layer=l
                    )
        return

    def _compute_feature_(self, getter, key=None, layer=None, pb=None):
        if key is None:
            return

        dataset = self.datasets[key]
        network = dataset.networks[0].data
        inputs = dataset.inputs[0].data[self.indices[key]]

        features = getter(inputs, network, layer=layer)
        return features

    def _get_indices_(self, dataset, doall=False):
        inputs = dataset.inputs[0].data
        T = inputs.shape[0]
        all_indices = np.arange(T)
        if not doall:
            weights = dataset.state_weights[0].data
            all_indices = all_indices[weights > 0]
        num_points = min(T, self.max_num_points)
        return np.random.choice(all_indices, size=num_points, replace=False)

    def _get_attcoeffs_(self, inputs, network, layer=None):

        gnn = self.model.nn.gnn_layer
        edge_index, edge_attr, node_attr = self.model.nn.transformers[
            "t_networks"
        ].forward(network)

        if layer is not None and isinstance(network, MultiplexNetwork):
            edge_index, edge_attr = edge_index[layer], edge_attr[layer]
            gnn = getattr(gnn, f"layer_{layer}")

        if node_attr is not None:
            node_attr = self.model.nn.node_layers(node_attr)
        if edge_attr is not None:
            edge_attr = self.model.nn.edge_layers(edge_attr)

        results = np.zeros((inputs.shape[0], edge_index.shape[1], gnn.heads))
        T, M = inputs.shape[0], edge_attr.shape[0]
        for i, x in enumerate(inputs):
            x = self.model.nn.transformers["t_inputs"].forward(x)
            x = self.model.nn.in_layers(x)
            x = self.model.nn.merge_nodeattr(x, node_attr)
            out = gnn.forward(x, edge_index, edge_attr, return_attention_weights=True)
            results[i] = out[1][1].detach().cpu().numpy()
        return results.reshape(T * M, -1)

    def _get_states_(self, inputs, network, layer=None):

        node_attr = network.node_attr
        edge_attr = network.edge_attr
        edge_index = network.edges

        if layer is not None and isinstance(network, MultiplexNetwork):
            edge_index, edge_attr = edge_index[layer], edge_attr[layer]

        T, M = inputs.shape[0], edge_index.shape[0]
        results = np.zeros((inputs.shape[0], *edge_index.shape, self.model.num_states))
        for i, x in enumerate(inputs):
            s, t = edge_index.T
            x = x.T[-1].T
            sources, targets = np.expand_dims(x[s], 1), np.expand_dims(x[t], 1)

            results[i] = np.concatenate((sources, targets), axis=1)
        return results.reshape(T * M, 2, -1)

    def _get_nodeattr_(self, inputs, network, layer=None):
        node_attr = network.node_attr
        edge_attr = network.edge_attr
        edge_index = network.edges

        if layer is not None and isinstance(network, MultiplexNetwork):
            edge_index, edge_attr = edge_index[layer], edge_attr[layer]

        T, M = inputs.shape[0], edge_index.shape[0]
        s, t = edge_index.T
        results = {}
        for k, v in node_attr.items():
            sources, targets = np.expand_dims(v[s], 1), np.expand_dims(v[t], 1)
            r = np.concatenate((sources, targets), axis=1)
            results[k] = r.reshape(1, *r.shape).repeat(T, axis=0).reshape(T * M, 2, -1)

        return results

    def _get_edgeattr_(self, inputs, network, layer=None):
        node_attr = network.node_attr
        edge_attr = network.edge_attr
        edge_index = network.edges

        if layer is not None and isinstance(network, MultiplexNetwork):
            edge_index, edge_attr = edge_index[layer], edge_attr[layer]

        T, M = inputs.shape[0], edge_index.shape[0]

        results = {
            k: v.reshape(1, *v.shape).repeat(T, axis=0).reshape(T * M, -1)
            for k, v in edge_attr.items()
        }

        return results


class AttentionNMIMetrics(AttentionMetrics):
    def __init__(self, config):
        AttentionMetrics.__init__(self, config)
        self.__attcoeffs_entropy = None
        self.__state_entropy = None
        self.__nodeattr_entropy = None
        self.__edgeattr_entropy = None
        p = config.__dict__.copy()
        self.n_neighbors = p.get("n_neighbors", 3)
        self.metric = p.get("metric", "euclidean")

    def initialize(self, experiment):
        self.model = experiment.model
        self.datasets = {
            "all": experiment.dataset,
            "train": experiment.dataset,
            "val": experiment.val_dataset,
            "test": experiment.test_dataset,
        }
        for n, d in self.datasets.items():
            if d is None:
                continue
            elif n == "all":
                self.indices[n] = self._get_indices_(d, doall=True)
            else:
                self.indices[n] = self._get_indices_(d, doall=False)
        if isinstance(self.model.nn.gnn_layer, DynamicsGATConv):
            if self.model.config.is_multiplex:
                self.layers = layer = self.model.config.network_layers
            else:
                layers = [None]
            for l in layers:
                name = "nmi"
                if l is not None:
                    name += f"-{l}"
                for n, d in self.datasets.items():
                    if d is None:
                        continue
                    attcoeffs = self._compute_feature_(
                        self._get_attcoeffs_, key=n, layer=l
                    )  # (T, M, H)
                    states = self._compute_feature_(
                        self._get_states_, key=n, layer=l
                    )  # (T, M, 2, D)
                    node_attr = self._compute_feature_(
                        self._get_nodeattr_, key=n, layer=l
                    )  # (T, M, 2, D)
                    edge_attr = self._compute_feature_(
                        self._get_edgeattr_, key=n, layer=l
                    )  # (T, M, D)

                    self.names.append(f"{n}-" + name + "att_vs_states")
                    self.get_data[f"{n}-" + name + "att_vs_states"] = partial(
                        self._compute_nmi_,
                        attcoeffs,
                        states.reshape(states.shape[0], -1),
                        self.names[-1],
                    )
                    self.names.append(f"{n}-" + name + "att_vs_s-states")
                    self.get_data[f"{n}-" + name + "att_vs_s-states"] = partial(
                        self._compute_nmi_, attcoeffs, states[:, 0, :], self.names[-1]
                    )
                    self.names.append(f"{n}-" + name + "att_vs_t-states")
                    self.get_data[f"{n}-" + name + "att_vs_t-states"] = partial(
                        self._compute_nmi_, attcoeffs, states[:, 1, :], self.names[-1]
                    )
                    for k, na in node_attr.items():
                        if na.ndim == 2:
                            na = na.reshape(*na.shape, 1)
                        self.names.append(f"{n}-" + name + "att_vs_nodeattr-" + k)
                        self.get_data[
                            f"{n}-" + name + "att_vs_nodeattr-" + k
                        ] = partial(
                            self._compute_nmi_,
                            attcoeffs,
                            na.reshape(na.shape[0], -1),
                            self.names[-1],
                        )

                        x = na[:, 0, :]

                        self.names.append(f"{n}-" + name + "att_vs_s-nodeattr-" + k)
                        self.get_data[
                            f"{n}-" + name + "att_vs_s-nodeattr-" + k
                        ] = partial(
                            self._compute_nmi_, attcoeffs, na[:, 0, :], self.names[-1]
                        )

                        self.names.append(f"{n}-" + name + "att_vs_t-nodeattr-" + k)
                        self.get_data[
                            f"{n}-" + name + "att_vs_t-nodeattr-" + k
                        ] = partial(
                            self._compute_nmi_, attcoeffs, na[:, 1, :], self.names[-1]
                        )

                    for k, ea in edge_attr.items():
                        self.names.append(f"{n}-" + name + "att_vs_edgeattr-" + k)
                        self.get_data[
                            f"{n}-" + name + "att_vs_edgeattr-" + k
                        ] = partial(self._compute_nmi_, attcoeffs, ea, self.names[-1])
        return

    def _compute_nmi_(self, x, y, name, pb=None):
        mi = mutual_info(x, y, n_neighbors=self.n_neighbors, metric=self.metric)
        hx = mutual_info(x, x, n_neighbors=self.n_neighbors, metric=self.metric)
        hy = mutual_info(y, y, n_neighbors=self.n_neighbors, metric=self.metric)
        if hx == 0 and hy == 0:
            return 0.0
        return 2 * mi / (hx + hy)
