import torch
import numpy as np

from dynalearn.utilities import onehot


def weighted_cross_entropy(y_pred, y_true, weights=None):
    num_nodes = y_pred.size(0)
    if weights is None:
        weights = torch.ones([y_true.size(i) for i in range(y_true.dim() - 1)])
    if torch.cuda.is_available():
        y_pred = y_pred.cuda()
        y_true = y_true.cuda()
        weights = weights.cuda()
    weights /= weights.sum()
    y_pred = torch.clamp(y_pred, 1e-15, 1 - 1e-15)
    loss = weights * (-y_true * torch.log(y_pred)).sum(-1)
    return loss.sum()


def weighted_mse(y_pred, y_true, weights=None):
    num_nodes = y_pred.size(0)
    if weights is None:
        weights = torch.ones([y_true.size(i) for i in range(y_true.dim() - 1)])
    if torch.cuda.is_available():
        y_pred = y_pred.cuda()
        y_true = y_true.cuda()
        weights = weights.cuda()

    weights /= weights.sum()
    loss = weights * torch.sum((y_true - y_pred) ** 2, axis=-1)
    # if np.isnan(loss.sum().cpu().detach().numpy()):
    #     print("Nan encountered in the loss computation:")
    #     print("y_true:", y_true)
    #     print("y_pred:", y_pred)
    #     print("w:", w)
    #     print("loss:", loss)

    return loss.sum()


__losses__ = {
    "weighted_cross_entropy": weighted_cross_entropy,
    "weighted_mse": weighted_mse,
    "cross_entropy": torch.nn.CrossEntropyLoss(),
    "cross_entropy": torch.nn.MSELoss(),
}


def get(loss):
    if loss in __losses__:
        return __losses__[loss]
    else:
        raise ValueError(
            f"{name} is invalid, possible entries are {list(__losses__.keys())}"
        )
