import torch
import numpy as np

from dynalearn.utilities import onehot


def weighted_cross_entropy(y_pred, y_true, weights=None):
    num_nodes = y_pred.size(0)
    if y_pred.dim() == y_true.dim() + 1:
        y_true = onehot(y_true, num_class=y_pred.size(-1))
    if weights is None:
        weights = torch.ones([y_true.size(i) for i in range(y_true.dim() - 1)])
    weights /= weights.sum()
    loss = weights * (-y_true * torch.log(y_pred)).sum(-1)
    return loss.sum()


__losses__ = {
    "weighted_cross_entropy": weighted_cross_entropy,
    "cross_entropy": torch.nn.CrossEntropyLoss(),
}


def get(loss):
    if loss in __losses__:
        return __losses__[loss]
    else:
        raise ValueError(
            f"{name} is invalid, possible entries are {list(__losses__.keys())}"
        )
