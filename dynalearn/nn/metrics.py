import torch

from dynalearn.utilities import onehot
from .loss import weighted_cross_entropy

EPSILON = 1e-8


def model_entropy(y_pred, y_true, weights=None):
    y_pred = torch.clamp(y_pred, EPSILON, 1 - EPSILON)
    if weights is None:
        return torch.mean((y_pred * torch.log(y_pred)).sum(-1))
    else:
        weights /= weights.sum()
        return torch.sum(weights * (y_pred * torch.log(y_pred)).sum(-1))


def relative_entropy(y_pred, y_true, weights=None):
    if y_true.dim() + 1 == y_pred.dim():
        y_true = onehot(y_pred, y_true.size(-1))
    y_true = torch.clamp(y_true, EPSILON, 1 - EPSILON)
    y_pred = torch.clamp(y_pred, EPSILON, 1 - EPSILON)
    cross_entropy = weighted_cross_entropy(y_pred, y_true, weights=weights)
    entropy = weighted_cross_entropy(y_true, y_true, weights=weights)
    return cross_entropy - entropy


def approx_relative_entropy(y_pred, y_true, weights=None):
    if y_true.dim() + 1 == y_pred.dim():
        y_true = onehot(y_pred, y_true.size(-1))
    y_true = torch.clamp(y_true, EPSILON, 1 - EPSILON)
    y_pred = torch.clamp(y_pred, EPSILON, 1 - EPSILON)
    cross_entropy = weighted_cross_entropy(y_pred, y_true, weights=weights)
    entropy = weighted_cross_entropy(y_pred, y_pred, weights=weights)
    return cross_entropy - entropy


def jensenshannon(y_pred, y_true, weights=None):
    if y_true.dim() + 1 == y_pred.dim():
        y_true = onehot(y_true, y_pred.size(-1))
    y_true = torch.clamp(y_true, EPSILON, 1 - EPSILON)
    y_pred = torch.clamp(y_pred, EPSILON, 1 - EPSILON)
    m = 0.5 * (y_true + y_pred)
    return 0.5 * (relative_entropy(y_true, m) + relative_entropy(y_pred, m))


__metrics__ = {
    "model_entropy": model_entropy,
    "relative_entropy": relative_entropy,
    "approx_relative_entropy": approx_relative_entropy,
    "jensenshannon": jensenshannon,
}


def get(names):
    metrics = {}
    for n in names:
        if n in __metrics__:
            metrics[n] = __metrics__[n]
        else:
            raise ValueError(
                f"{name} is invalid, possible entries are {list(__metrics__.keys())}"
            )
    return metrics
