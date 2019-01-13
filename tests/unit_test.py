import torch
import numpy as np

import dynalearn as dl
from dynalearn.model.unit import *

if __name__ == '__main__':
    N = 100000
    mean = 2
    ui = Unit_info("v", N, "visible", "bernoulli")
    print(ui)
    u = Unit("v", ui, 1)
    print("Unit v: ", u.data)
    mean_arr = torch.ones(N) * mean
    print("Unit v (sampled): ", u.sample(mean=mean_arr))
    print("Activation probability of v: ", torch.exp(u.log_p(mean=mean_arr)))
    print("Mean v: ", u.data.mean())
    print("Mean v: ", 1 / (1 + np.exp(-mean)))
