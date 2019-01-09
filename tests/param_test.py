from dynalearn.model.param import *
import torch
from torch.autograd import backward

from dynalearn.model.unit import *


if __name__ == '__main__':
    v_i = Unit_info("v", 2, "visible", "bernoulli")
    h_i = Unit_info("h", 2, "hidden", "bernoulli")
    batchsize = 2
    v = Unit("v", v_i, batchsize)
    h = Unit("h", h_i, batchsize)
    units = {"v":v, "h":h}

    w = Weight([v, h], 0.01, False)
    w_v = Weight([v, v], 0.01, False)
    w_h = Weight([h, h], 0.01, False)
    b = Bias(v, None, False)
    c = Bias(h, None, False)

    e = - w.energy_term(units)\
        - w_v.energy_term(units)\
        - w_h.energy_term(units)\
        - b.energy_term(units)\
        - c.energy_term(units)

    e.backward()

    print("Visible : ", units["v"].data)
    print("Hidden : ", units["h"].data)
    print("Gradient of W: ", w.param.grad) 
    print("Gradient of W_v: ", w_v.param.grad) 
    print("Gradient of W_h: ", w_h.param.grad) 
    print("Gradient of b: ", b.param.grad) 
    print("Gradient of c: ", c.param.grad) 

    # params = {'vh': w, 'v': b, 'h': c}
    # print(params)
    # params = Param(params)

    # for k in params._modules:
    #     print(k)
