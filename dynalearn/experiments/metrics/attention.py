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
        self.dataset = experiment.dataset
        self.indices = self._get_indices_()

        if isinstance(self.model.nn.gnn_layer, DynamicsGATConv):
            if self.model.config.is_multiplex:
                self.layers = layer = self.model.config.network_layers
            else:
                layers = [None]
            for l in layers:
                name = "attcoeffs"
                if l is not None:
                    name += f"-{l}"
                self.names.append(name)
                self.get_data[name] = partial(self._get_attcoeffs_, layer=l)
        return

    def _get_indices_(self, doall=True):
        inputs = self.dataset.inputs[0].data
        T = inputs.shape[0]
        all_indices = np.arange(T)
        if not doall:
            weights = self.dataset.state_weights[0].data
            all_indices = all_indices[weights > 0]
        num_points = min(T, self.max_num_points)
        return np.random.choice(all_indices, size=num_points, replace=False)

    def _get_attcoeffs_(self, layer=None, pb=None):
        network = self.dataset.networks[0].data
        inputs = self.dataset.inputs[0].data[self.indices]
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


class AttentionFeatureNMIMetrics(AttentionMetrics):
    def __init__(self, config):
        AttentionMetrics.__init__(self, config)
        p = config.__dict__.copy()
        self.n_neighbors = p.get("n_neighbors", 3)
        self.metric = p.get("metric", "euclidean")
        self.fname = None

    def initialize(self, experiment):
        self.model = experiment.model
        self.dataset = experiment.dataset
        self.indices = self._get_indices_()
        if isinstance(self.model.nn.gnn_layer, DynamicsGATConv):
            if self.model.config.is_multiplex:
                self.layers = layer = self.model.config.network_layers
            else:
                layers = [None]
            for l in layers:
                name = "nmi-"
                if l is not None:
                    name = f"{l}-" + name
                attcoeffs = self._get_attcoeffs_(layer=l)
                features = self._get_feature_(layer=l)
                if isinstance(features, dict):
                    for k, v in features.items():
                        d_name = name + f"att_vs_{self.fname}-{k}"
                        self.names.append(d_name)
                        self.get_data[d_name] = partial(
                            self._compute_nmi_, attcoeffs, v
                        )
        return

    def _get_feature_(self, inputs, network, layer=None):
        raise notImplemented

    def _compute_nmi_(self, x, y, pb=None):
        mi = mutual_info(x, y, n_neighbors=self.n_neighbors, metric=self.metric)
        hx = mutual_info(x, x, n_neighbors=self.n_neighbors, metric=self.metric)
        hy = mutual_info(y, y, n_neighbors=self.n_neighbors, metric=self.metric)
        if hx == 0 and hy == 0:
            return 0.0
        return 2 * mi / (hx + hy)


class AttentionStatesNMIMetrics(AttentionFeatureNMIMetrics):
    def __init__(self, config):
        AttentionFeatureNMIMetrics.__init__(self, config)
        self.fname = "states"

    def _get_feature_(self, layer=None):

        network = self.dataset.networks[0].data
        inputs = self.dataset.inputs[0].data[self.indices]
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
        results = results.reshape(T * M, 2, -1)

        return {
            "all": results.reshape(T * M, -1),
            "source": results[:, 0, :],
            "target": results[:, 1, :],
        }


class AttentionNodeAttrNMIMetrics(AttentionFeatureNMIMetrics):
    def __init__(self, config):
        AttentionFeatureNMIMetrics.__init__(self, config)
        self.fname = "nodeattr"

    def _get_feature_(self, layer=None):

        network = self.dataset.networks[0].data
        inputs = self.dataset.inputs[0].data[self.indices]
        node_attr = network.node_attr
        edge_attr = network.edge_attr
        edge_index = network.edges

        if layer is not None and isinstance(network, MultiplexNetwork):
            edge_index, edge_attr = edge_index[layer], edge_attr[layer]

        T, M = inputs.shape[0], edge_index.shape[0]
        s, t = edge_index.T
        res = {}
        for k, v in node_attr.items():
            sources, targets = np.expand_dims(v[s], 1), np.expand_dims(v[t], 1)
            r = np.concatenate((sources, targets), axis=1)
            res[k] = r.reshape(1, *r.shape).repeat(T, axis=0).reshape(T * M, 2, -1)
        results = {"all-" + k: v.reshape(T * M, -1) for k, v in res.items()}
        results.update({"source-" + k: v[:, 0, :] for k, v in res.items()})
        results.update({"target-" + k: v[:, 1, :] for k, v in res.items()})
        return results


class AttentionEdgeAttrNMIMetrics(AttentionFeatureNMIMetrics):
    def __init__(self, config):
        AttentionFeatureNMIMetrics.__init__(self, config)
        self.fname = "edgeattr"

    def _get_feature_(self, layer=None):

        network = self.dataset.networks[0].data
        inputs = self.dataset.inputs[0].data[self.indices]
        node_attr = network.node_attr
        edge_attr = network.edge_attr
        edge_index = network.edges

        if layer is not None and isinstance(network, MultiplexNetwork):
            edge_index, edge_attr = edge_index[layer], edge_attr[layer]

        T, M = inputs.shape[0], edge_index.shape[0]
        s, t = edge_index.T
        results = {}
        for k, v in edge_attr.items():
            results[k] = v.reshape(1, *v.shape).repeat(T, axis=0).reshape(T * M, -1)
        return results
