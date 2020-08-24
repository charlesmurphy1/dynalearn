import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
from numba import jit
from cmath import exp, log
from math import log as mlog

color_dark = {
    "blue": "#1f77b4",
    "orange": "#f19143",
    "purple": "#9A80B9",
    "red": "#d73027",
    "grey": "#525252",
    "green": "#33b050",
}

color_pale = {
    "blue": "#7bafd3",
    "orange": "#f7be90",
    "purple": "#c3b4d6",
    "red": "#e78580",
    "grey": "#999999",
    "green": "#9fdaac",
}

colormap = "bone"

m_list = ["o", "s", "v", "^"]
l_list = ["solid", "dashed", "dotted", "dashdot"]
cd_list = [
    color_dark["blue"],
    color_dark["orange"],
    color_dark["purple"],
    color_dark["red"],
]
cp_list = [
    color_pale["blue"],
    color_pale["orange"],
    color_pale["purple"],
    color_pale["red"],
]

plt.rc("text", usetex=True)
plt.rc("font", family="serif")


def sigmoid(x):
    return 1.0 / (np.exp(-x) + 1)


def from_binary(x):
    n = np.arange(x.shape[0])[::-1]
    return (x * 2 ** (n)).sum()


def to_binary(x, max_val=None):
    max_val = max_val or 2 ** np.floor(np.log2(x) + 1)
    r = np.zeros(np.log2(max_val).astype("int"))
    r0 = x
    while r0 > 0:
        y = np.floor(np.log2(r0)).astype("int")
        r[y] = 1
        r0 -= 2 ** y
    return r[::-1]


def logbase(x, base=np.e):
    return np.log(x) / np.log(base)


def from_nary(x, axis=0, base=2):
    if type(x) is int or type(x) is float:
        x = np.array([x])
    n = np.arange(x.shape[axis])[::-1]
    n = n.reshape(*[s if i == axis else 1 for i, s in enumerate(x.shape)])
    return (x * base ** (n)).sum(axis)


def to_nary(x, base=2, dim=None):
    if type(x) is int or type(x) is float:
        x = np.array([x])
    if dim is None:
        max_val = base ** np.floor(logbase(np.max(x), base) + 1)
        dim = int(logbase(max_val, base))
    y = np.zeros([dim, *x.shape])
    for idx, xx in np.ndenumerate(x):
        r = np.zeros(dim)
        r0 = xx
        while r0 > 0:
            b = int(np.floor(logbase(r0, base)))
            r[b] += 1
            r0 -= base ** b
        y.T[idx] = r[::-1]
    return y


def all_combinations(n, k):
    t = n
    h = 0
    a = [0] * k
    a[0] = n
    res = []
    res.append(a.copy())
    while a[k - 1] != n:
        if t != 1:
            h = 0
        t = a[h]
        a[h] = 0
        a[0] = t - 1
        a[h + 1] += 1
        h += 1
        res.append(a.copy())
    return res


@jit(nopython=True)
def numba_all_combinations(n, k):
    t = n
    h = 0
    a = [0] * k
    a[0] = n
    res = []
    res.append(a.copy())
    while a[k - 1] != n:
        if t != 1:
            h = 0
        t = a[h]
        a[h] = 0
        a[0] = t - 1
        a[h + 1] += 1
        h += 1
        res.append(a.copy())
    return res


@jit(nopython=True)
def numba_factorial(k):
    res = 1
    for i in range(k):
        res *= i + 1
    return res


@jit(nopython=True)
def numba_logfactorial(k):
    res = 0
    for i in range(k):
        res += np.log(i + 1)
    return res


@jit(nopython=True)
def numba_multinomial(k, l, phi):
    p = numba_factorial(k)
    for i in range(len(phi)):
        p *= phi[i] ** l[i] / numba_factorial(l[i])
    return p


def k_l_grid(k_arr, l_arr, s_dim):
    neigh_array = np.meshgrid(k_arr, *[l_arr] * s_dim)
    k_grid = neigh_array[0]
    l_grid = np.zeros((s_dim, len(k_arr), *[len(l_arr)] * s_dim))
    _l_grid = [neigh_array[i + 1] for i in range(s_dim)]

    ind = np.arange(s_dim + 1)
    ind[0] = 1
    ind[1] = 0
    k_grid = k_grid.transpose(ind)
    for i in range(s_dim):
        l_grid[i] = _l_grid[i].transpose(ind)
    return k_grid, l_grid


def onehot(x, num_class=None, dim=-1):
    if type(x) == np.ndarray:
        return onehot_numpy(x, num_class, dim)
    elif type(x) == torch.Tensor:
        return onehot_torch(x, num_class, dim)
    else:
        raise ValueError(
            f"{type(x)} is an invalid type, valid types are [np.array, torch.Tensor]"
        )


def onehot_torch(x, num_class=None, dim=-1):
    if num_class is None:
        num_class = num_class or int(x.max()) + 1
    x_onehot = torch.zeros(*tuple(x.size()), num_class).float()
    if torch.cuda.is_available():
        x_onehot = x_onehot.cuda()
    x = x.long().view(-1, 1)
    x_onehot.scatter_(dim, x, 1)
    return x_onehot


def onehot_numpy(x, num_class=None, dim=-1):
    num_class = num_class or int(x.max()) + 1
    y = np.zeros((*x.shape, num_class))
    i_shape = x.shape
    x = x.reshape(-1)
    y = y.reshape(-1, num_class)
    y[np.arange(x.size), x.astype("int")] = 1
    y = y.reshape((*i_shape, num_class))
    return y


def to_edge_index(g):
    if not nx.is_directed(g):
        g = g.to_directed()

    if len(list(g.edges())) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = np.array(list(nx.to_edgelist(g)))[:, :2].astype("int").T
        edge_index = torch.LongTensor(edge_index)

    return edge_index


def collapse_networks(g_dict):
    if not isinstance(g_dict, dict):
        return g_dict
    g = nx.empty_graph()
    for k, v in g_dict.items():
        edge_list = to_edge_index(v).numpy().T
        edge_attr = get_edge_attr(v)
        g.add_edges_from(edge_list)
        for i, (u, v) in enumerate(edge_list):
            if "weight" in g.edges[u, v]:
                g.edges[u, v]["weight"] += edge_attr["weight"][i]
            else:
                g.edges[u, v]["weight"] = edge_attr["weight"][i]
    return g


def get_edge_weights(g):
    if not nx.is_directed(g):
        g = g.to_directed()

    edge_index = to_edge_index(g).numpy().T
    weights = np.zeros((edge_index.shape[0], 1))

    for i, (u, v) in enumerate(edge_index):
        if "weight" in g.edges[u, v]:
            weights[i] = g.edges[u, v]["weight"]
        else:
            weights[i] = 1

    return weights


def get_edge_attr(g, to_data=False):
    if not nx.is_directed(g):
        g = g.to_directed()

    edge_index = to_edge_index(g).numpy().T
    attributes = {}

    for i, (u, v) in enumerate(edge_index):
        attr = g.edges[u, v]
        for k, a in attr.items():
            if k not in attributes:
                attributes[k] = np.zeros(edge_index.shape[0])
            attributes[k][i] = a
    if to_data:
        return np.concatenate(
            [v.reshape(-1, 1) for k, v in attributes.items()], axis=-1,
        )
    return attributes


def set_edge_attr(g, edge_attr_dict):

    edge_index = to_edge_index(g).numpy().T
    for k, attr in edge_attr_dict.items():
        for i, (u, v) in enumerate(edge_index):
            g.edges[u, v][k] = attr[i]
    return g


def get_node_strength(g):
    if not nx.is_directed(g):
        g = g.to_directed()

    strength = np.zeros(g.number_of_nodes())

    for u, v in g.edges():
        if "weight" in g.edges[u, v]:
            strength[u] += g.edges[u, v]["weight"]

    return strength


def from_weighted_edgelist(edge_list, create_using=None):
    g = create_using or nx.Graph()
    for edge in edge_list:
        if len(edge) == 3:
            g.add_edge(int(edge[0]), int(edge[1]), weight=edge[2])
        else:
            g.add_edge(int(edge[0]), int(edge[1]), weight=1)
    return g
