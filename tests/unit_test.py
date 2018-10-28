import torch

import dynalearn as dl
from dynalearn.model.unit import *

if __name__ == '__main__':
    ui = Unit_info("v", 10, "visible", "bernoulli")
    print(ui)
    u = Unit("v", ui, 1)
    print("Unit v: ", u.data)
    p = torch.ones(10) * (-1e10)
    print("Unit v (sampled): ", u.sample(mean=p))
    print("Activation probability of v: ", torch.exp(u.log_p(mean=p)))
