import torch
import numpy as np
import math

from dynalearn.model.param import *
from dynalearn.model.unit import *
from dynalearn.model.crbm import *
from dynalearn.utilities.utilities import *


if __name__ == '__main__':
    
    n_visible = 5
    n_conditional = 4
    n_hidden = 3
    batchsize = 2

    rbm = CRBM_BernoulliBernoulli(n_visible, n_conditional, n_hidden)
    rbm.model_config.BATCHSIZE = batchsize
    
    units = rbm.init_units()
    rbm.compute_log_Z()
    v = units["v"].data

    print("Sampling : ", rbm(v, 10))
    print("Energy : ", rbm.energy(units))
    print("Free energy : ", rbm.free_energy(v))
    print("Reconstruction : ", rbm.reconstruction(v))
    print("Inference : ", rbm.inference(v))
    print("Conditional log-p : ", rbm.conditional_log_p(v))
    print("Reconstruction MSE : ", rbm.reconstruction_MSE(v))
    # print("Partition function : ", rbm.comput(v))
    # print("Likelihood : ", rbm.log_likelihood(v))

