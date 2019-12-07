import numpy as np

EPSILON = 1e-10


def config_k_l_grid(k_arr, l_arr, s_dim):
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


def marginalize_on_l(x, s_dim, bad_config):
    _x = x.copy()
    _x.T[bad_config.T] = 0
    for i in range(s_dim):
        _x = _x.sum(-1)
    return _x


def power_method(A, tol=1e-5, max_iter=100):
    x0 = np.random.rand(A.shape[1])
    diff = np.inf
    it = 0
    while diff > tol:
        x = np.dot(A, x0)
        x = x / np.linalg.norm(x)
        diff = np.linalg.norm(x - x0)
        x0 = x
        it += 0
        if it > max_iter:
            print("Not much progress is being made.")
            break
    val = x.T @ A @ x / (x ** 2).sum()
    vec = x
    return val, vec
