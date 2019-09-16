import numpy as np


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
