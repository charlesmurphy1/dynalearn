import numpy as np
import torch

from .transformer import Transformer, CUDATransformer, IdentityTransformer
from dynalearn.utilities import to_edge_index, get_edge_attr, get_node_attr


class Normalizer(Transformer):
    def __init__(self, name, shape=(), axis=0, auto_cuda=True):
        Transformer.__init__(self, name)
        self.axis = axis
        if isinstance(shape, int):
            if shape > 0:
                self.shape = (shape,)
            else:
                self.shape = ()
        elif isinstance(shape, (list, tuple)):
            self.shape = tuple(shape)
        else:
            raise ValueError(f"{self.shape} is an invalid shape.")

        if len(self.shape) > 0:
            self.register_buffer(
                f"{self.name}_mean", torch.zeros(torch.Size(self.shape))
            )
            self.register_buffer(f"{self.name}_var", torch.ones(torch.Size(self.shape)))
            self.is_empty = False
        else:
            self.register_buffer(f"{self.name}_mean", None)
            self.register_buffer(f"{self.name}_var", None)
            self.is_empty = True

        if auto_cuda:
            self.cuda_transformer = CUDATransformer()
        else:
            self.cuda_transformer = IdentityTransformer()

    def getter(self, index, dataset):
        raise NotImplemented()

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        x = self.cuda_transformer.forward(x)
        m = getattr(self, f"{self.name}_mean")
        v = getattr(self, f"{self.name}_var")
        if self.is_empty:
            y = x
        else:
            y = (x - m) / v ** (0.5)
        return self.cuda_transformer.forward(y)

    def backward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        x = self.cuda_transformer.forward(x)

        m = getattr(self, f"{self.name}_mean")
        v = getattr(self, f"{self.name}_var")
        if self.is_empty:
            y = x
        else:
            y = x * v ** (0.5) + m
        return self.cuda_transformer.backward(y)

    def _setUp_mean(self, dataset):
        return self._normalizer_template_(dataset, torch.mean)

    def _setUp_var(self, dataset):
        return self._normalizer_template_(dataset, torch.var)

    def _normalizer_template_(self, dataset, operator):
        y = None
        if self.is_empty:
            return y
        for i in range(dataset.networks.size):
            x = self.getter(i, dataset)
            assert isinstance(x, torch.Tensor)
            if x.numel() == 0:
                self.is_empty = True
                return y
            if y is None:
                y = operator(x, dim=self.axis, keepdims=True) / dataset.networks.size
            else:
                y += operator(x, dim=self.axis, keepdims=True) / dataset.networks.size

        return y.view(self.shape)


class InputNormalizer(Normalizer):
    def __init__(self, size, auto_cuda=True):
        self.size = size
        if size > 0:
            shape = (1, size, 1)
        else:
            shape = ()
        axis = (0, 1, 2)
        Normalizer.__init__(self, "inputs", shape=shape, axis=axis, auto_cuda=auto_cuda)

    def getter(self, index, dataset):
        return torch.Tensor(dataset.inputs[index].data)


class TargetNormalizer(Normalizer):
    def __init__(self, size, auto_cuda=True):
        self.size = size
        if size > 0:
            shape = (1, size)
        else:
            shape = ()
        axis = (0, 1)
        Normalizer.__init__(
            self, "targets", shape=shape, axis=axis, auto_cuda=auto_cuda
        )

    def getter(self, index, dataset):
        return torch.Tensor(dataset.targets[index].data)


class NodeNormalizer(Normalizer):
    def __init__(self, size, layer=None, auto_cuda=True):
        self.size = size
        self.layer = layer
        if self.layer is None:
            label = ""
        else:
            label = f"_{self.layer}"
        if size > 0:
            shape = (1, size)
        else:
            shape = ()
        Normalizer.__init__(
            self, f"nodeattr{label}", shape=shape, axis=0, auto_cuda=auto_cuda,
        )

    def getter(self, index, dataset):
        if self.layer is None:
            x = get_node_attr(dataset.networks[index].data, to_data=True)
        else:
            x = get_node_attr(dataset.networks[index].data[self.layer], to_data=True)

        return torch.Tensor(x)


class EdgeNormalizer(Normalizer):
    def __init__(self, size, layer=None, auto_cuda=True):
        self.size = size
        self.layer = layer
        if self.layer is None:
            label = ""
        else:
            label = f"_{self.layer}"
        if size > 0:
            shape = (1, size)
        else:
            shape = ()
        Normalizer.__init__(
            self, f"edgeattr{label}", shape=shape, axis=0, auto_cuda=auto_cuda,
        )

    def getter(self, index, dataset):
        if self.layer is None:
            x = get_edge_attr(dataset.networks[index].data, to_data=True)
        else:
            x = get_edge_attr(dataset.networks[index].data[self.layer], to_data=True)

        return torch.Tensor(x)


class NetworkNormalizer(Transformer):
    def __init__(self, node_size, edge_size, layers=None, auto_cuda=True):
        Transformer.__init__(self, "networks")
        self.node_size = node_size
        self.edge_size = edge_size
        self.layers = layers
        self.t_cuda = CUDATransformer()
        if layers is not None:
            for l in layers:
                setattr(
                    self, f"t_nodeattr_{l}", NodeNormalizer(layer=l, size=node_size)
                )
                setattr(
                    self, f"t_edgeattr_{l}", EdgeNormalizer(layer=l, size=edge_size)
                )
        else:
            setattr(self, "t_nodeattr", NodeNormalizer(size=node_size))
            setattr(self, "t_edgeattr", EdgeNormalizer(size=edge_size))

    def setUp(self, dataset):

        if self.layers is not None:
            for l in self.layers:
                getattr(self, f"t_nodeattr_{l}").setUp(dataset)
                getattr(self, f"t_edgeattr_{l}").setUp(dataset)
        else:
            getattr(self, "t_nodeattr").setUp(dataset)
            getattr(self, "t_edgeattr").setUp(dataset)

    def forward(self, g):
        if isinstance(g, dict):
            edge_index, edge_attr, node_attr = {}, {}, {}
            for k in self.layers:
                assert k in g.keys(), f"{k} is not a layer of the graph"
                edge_index[k], edge_attr[k], node_attr[k] = self._normalize_network_(
                    g[k], layer=k
                )
        else:
            edge_index, edge_attr, node_attr = self._normalize_network_(g)
        _g = (edge_index, edge_attr, node_attr)
        return _g

    def backward(self, g):
        return g

    def _normalize_network_(self, g, layer=None):
        e_key = "t_edgeattr"
        n_key = "t_nodeattr"
        if layer is not None:
            e_key += f"_{layer}"
            n_key += f"_{layer}"
        edge_index = self.t_cuda.forward(torch.LongTensor(to_edge_index(g)))
        node_attr = getattr(self, n_key).forward(
            torch.Tensor(get_node_attr(g, to_data=True))
        )

        edge_attr = getattr(self, e_key).forward(
            torch.Tensor(get_edge_attr(g, to_data=True))
        )
        if edge_attr.numel() == 0:
            edge_attr = None

        if node_attr.numel() == 0:
            node_attr = None
        return edge_index, edge_attr, node_attr
