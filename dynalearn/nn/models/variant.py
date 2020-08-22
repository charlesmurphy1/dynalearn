import torch

from .gnn import GraphNeuralNetwork
from dynalearn.utilities import to_edge_index, get_edge_attr


class ContinuousGraphNeuralNetwork(GraphNeuralNetwork):
    def __init__(self, config=None, **kwargs):
        torch.nn.Module.__init__(self)
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        GraphNeuralNetwork.__init__(self, config=config, **kwargs)
        self.input_shape = config.input_shape
        self.target_shape = config.target_shape
        self.register_buffer("_data_mean_inputs", torch.Tensor(self.input_shape))
        self.register_buffer("_data_var_inputs", torch.Tensor(self.input_shape))
        self.register_buffer("_data_mean_targets", torch.Tensor(self.target_shape))
        self.register_buffer("_data_var_targets", torch.Tensor(self.target_shape))

    def setUp(self, dataset):
        self._data_mean = {}
        self._data_var = {}
        self._setup_input_(dataset)
        self._setup_target_(dataset)

    def _setup_input_(self, dataset):
        for i in range(dataset.networks.size):
            x = torch.Tensor(dataset.inputs[i].data)
            if self.using_log:
                torch.clamp_(x, 1e-15)
                x = torch.log(x)

            if "inputs" not in self._data_mean:
                self._data_mean["inputs"] = (
                    torch.mean(x, (0, 1, 3)) / dataset.networks.size
                ).view(1, -1, 1)
                self._data_var["inputs"] = (
                    torch.var(x, (0, 1, 3)) / dataset.networks.size
                ).view(1, -1, 1)
            else:
                self._data_mean["inputs"] += (
                    torch.mean(x, (0, 1, 3)) / dataset.networks.size
                ).view(1, -1, 1)
                self._data_var["inputs"] += (
                    torch.var(x, (0, 1, 3)) / dataset.networks.size
                ).view(1, -1, 1)
        self._data_mean_inputs = self._data_mean["inputs"]
        self._data_var_inputs = self._data_var["inputs"]

    def _setup_target_(self, dataset):
        for i in range(dataset.networks.size):
            y = torch.Tensor(dataset.targets[i].data)
            if self.using_log:
                torch.clamp_(y, 1e-15)
                y = torch.log(y)

            if "targets" not in self._data_mean:
                self._data_mean["targets"] = (
                    torch.mean(y, (0, 1)) / dataset.networks.size
                ).view(1, -1)
                self._data_var["targets"] = (
                    torch.var(y, (0, 1)).view(1, -1) / dataset.networks.size
                )
            else:
                self._data_mean["targets"] += (
                    torch.mean(y, (0, 1)) / dataset.networks.size
                ).view(1, -1)
                self._data_var["targets"] += (
                    torch.var(y, (0, 1)) / dataset.networks.size
                ).view(1, -1)

        self._data_mean_targets = self._data_mean["targets"]
        self._data_var_targets = self._data_var["targets"]


class WeightedGraphNeuralNetwork(GraphNeuralNetwork):
    def __init__(self, config=None, **kwargs):
        torch.nn.Module.__init__(self)
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        GraphNeuralNetwork.__init__(self, config=config, **kwargs)
        self.edgeattr_shape = config.edgeattr_shape
        self.register_buffer("_data_mean_edgeattr", torch.Tensor(self.edgeattr_shape))
        self.register_buffer("_data_var_edgeattr", torch.Tensor(self.edgeattr_shape))

    def get_output(self, data):
        (x, g), y_true, w = data
        edge_index = to_edge_index(g)
        edge_attr = torch.Tensor(get_edge_attr(g, to_data=True))
        if torch.cuda.is_available():
            x = x.cuda()
            edge_index = edge_index.cuda()
            edge_attr = edge_attr.cuda()
            y_true = y_true.cuda()
            w = w.cuda()
        x = self.normalize(x, "inputs")
        y_true = self.normalize(y_true, "targets")
        edge_attr = self.normalize(edge_attr, "edgeattr")
        y = self.forward(x, edge_index, edge_attr=edge_attr)
        return y, y_true

    def setUp(self, dataset):
        self._data_mean = {}
        self._data_var = {}
        self._setup_edge_attr_(dataset)

    def _setup_edge_attr_(self, dataset):
        for i in range(dataset.networks.size):
            g = dataset.networks[i].data
            ew = torch.Tensor(get_edge_attr(g, to_data=True))
            if self.using_log:
                torch.clamp_(ew, 1e-15)
                ew = torch.log(ew)
            if "edgeattr" not in self._data_mean:
                self._data_mean["edgeattr"] = torch.mean(ew, 0) / dataset.networks.size
                self._data_var["edgeattr"] = torch.var(ew, 0) / dataset.networks.size
            else:
                self._data_mean["edgeattr"] += torch.mean(ew, 0) / dataset.networks.size
                self._data_var["edgeattr"] += torch.var(ew, 0) / dataset.networks.size

        self._data_mean_edgeattr = self._data_mean["edgeattr"]
        self._data_var_edgeattr = self._data_var["edgeattr"]


class MultiplexGraphNeuralNetwork(GraphNeuralNetwork):
    def __init__(self, config=None, **kwargs):
        GraphNeuralNetwork.__init__(self, config=config, **kwargs)
        self.network_layers = config.network_layers

    def get_output(self, data):
        (x, g), y_true, w = data
        edge_index = {k: to_edge_index(v) for k, v in g.items()}
        if torch.cuda.is_available():
            x = x.cuda()
            edge_index = {k: v.cuda() for k, v in edge_index.items()}
            y_true = y_true.cuda()
            w = w.cuda()
        x = self.normalize(x, "inputs")
        y_true = self.normalize(y_true, "targets")
        y = self.forward(x, edge_index, edge_attr=edge_attr)
        return y, y_true


class WeightedMultiplexGraphNeuralNetwork(GraphNeuralNetwork):
    def __init__(self, config=None, **kwargs):
        GraphNeuralNetwork.__init__(self, config=config, **kwargs)
        self.network_layers = config.network_layers
        self.edgeattr_shape = config.edgeattr_shape
        for k in self.network_layers:
            self.register_buffer(
                f"_data_mean_edgeattr_{k}", torch.Tensor(self.edgeattr_shape)
            )

    def get_output(self, data):
        (x, g), y_true, w = data
        edge_index = {k: to_edge_index(v) for k, v in g.items()}
        edge_attr = {
            k: torch.Tensor(get_edge_attr(v, to_data=True)) for k, v in g.items()
        }
        if torch.cuda.is_available():
            x = x.cuda()
            edge_index = {k: v.cuda() for k, v in edge_index.items()}
            edge_attr = {k: v.cuda() for k, v in edge_attr.items()}
            y_true = y_true.cuda()
            w = w.cuda()
        x = self.normalize(x, "inputs")
        y_true = self.normalize(y_true, "targets")
        edge_attr = self.normalize(edge_attr, "edgeattr")
        y = self.forward(x, edge_index, edge_attr=edge_attr)
        return y, y_true

    def setUp(self, dataset):
        self._data_mean = {}
        self._data_var = {}
        self._setup_edge_attr_(dataset)

    def _setup_edge_attr_(self, dataset):
        for i in range(dataset.networks.size):
            g = dataset.networks[i].data
            for k in self.network_layers:
                ew = torch.Tensor(get_edge_attr(g[v], to_data=True))
                if "edgeattr" not in self._data_mean:
                    self._data_mean["edgeattr"] = {
                        k: (torch.mean(ew, 0) / dataset.networks.size)
                    }
                    self._data_var["edgeattr"] = {
                        k: (torch.var(ew, 0) / dataset.networks.size)
                    }
                elif k not in self._data_mean["edgeattr"]:
                    self._data_mean["edgeattr"][k] = (
                        torch.mean(ew, 0) / dataset.networks.size
                    )
                    self._data_var["edgeattr"][k] = (
                        torch.var(ew, 0) / dataset.networks.size
                    )
                else:
                    self._data_mean["edgeattr"][k] += (
                        torch.mean(ew, 0) / dataset.networks.size
                    )
                    self._data_var["edgeattr"][k] += (
                        torch.var(ew, 0) / dataset.networks.size
                    )
        for k in self._data_mean["edgeattr"].keys():
            setattr(self, f"_data_mean_edgeattr_{k}", self._data_mean["edgeattr"][k])
            setattr(self, f"_data_var_edgeattr_{k}", self._data_var["edgeattr"][k])


class ContinuousWeightedGraphNeuralNetwork(
    ContinuousGraphNeuralNetwork, WeightedGraphNeuralNetwork
):
    def __init__(self, config=None, **kwargs):
        WeightedGraphNeuralNetwork.__init__(self, config=config, **kwargs)
        ContinuousGraphNeuralNetwork.__init__(self, config=config, **kwargs)

    def setUp(self, dataset):
        self._data_mean = {}
        self._data_var = {}
        self._setup_input_(dataset)
        self._setup_target_(dataset)
        self._setup_edge_attr_(dataset)


class ContinuousMultiplexGraphNeuralNetwork(
    ContinuousGraphNeuralNetwork, MultiplexGraphNeuralNetwork
):
    def __init__(self, config=None, **kwargs):
        ContinuousGraphNeuralNetwork.__init__(self, config=config, **kwargs)
        MultiplexGraphNeuralNetwork.__init__(self, config=config, **kwargs)


class ContinuousWeightedMultiplexGraphNeuralNetwork(
    ContinuousGraphNeuralNetwork, WeightedMultiplexGraphNeuralNetwork
):
    def __init__(self, config=None, **kwargs):
        ContinuousGraphNeuralNetwork.__init__(self, config=config, **kwargs)
        WeightedMultiplexGraphNeuralNetwork.__init__(self, config=config, **kwargs)

    def setUp(self, dataset):
        self._data_mean = {}
        self._data_var = {}
        self._setup_input_(dataset)
        self._setup_target_(dataset)
        self._setup_edge_attr_(dataset)
