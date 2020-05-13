import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
from numba import jit
from cmath import exp, log

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


def to_binary(x, max_val):
    r = np.zeros(np.log2(max_val).astype("int"))
    r0 = x
    while r0 > 0:
        y = np.floor(np.log2(r0)).astype("int")
        r[y] = 1
        r0 -= 2 ** y
    return r[::-1]


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
    if num_class is None:
        num_class = int(x.max())
    x_onehot = torch.zeros(*tuple(x.size()), num_class).float()
    if torch.cuda.is_available():
        x_onehot = x_onehot.cuda()
    x = x.long().view(-1, 1)
    x_onehot.scatter_(dim, x, 1)
    return x_onehot


def to_edge_index(g):
    g = g.to_directed()

    if len(list(g.edges())) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = np.array(list(nx.to_edgelist(g)))[:, :2].astype("int").T
        edge_index = torch.LongTensor(edge_index)

    return edge_index
