import copy
import torch
from dynalearn.model.param import *
from dynalearn.model.unit import *
from dynalearn.model.rbm import *
from dynalearn.model.config import *
from dynalearn.utilities.utilities import *

if __name__ == '__main__':
    
    n_visible = 3
    n_hidden = 3
    batchsize = 1

    config = Config(batchsize=batchsize, num_sample=10)
    rbm = RBM_BernoulliBernoulli(n_visible, n_hidden, config)
    # rbm.params['v'].init_value(0.1)

    w = rbm.params['vh'].param
    b = rbm.params['v'].param
    c = rbm.params['h'].param
    # for k in rbm.params:
        # print(k)
    
    units = rbm.init_units()
    rbm.compute_log_Z()
    v = units["v"].data

    e = rbm.energy(units)
    fe = rbm.free_energy(v)
    recon = rbm.reconstruction(v)
    inf = rbm.inference(v)
    log_Z = rbm.log_Z
    logp = rbm.log_likelihood(v)
    print("Sampling : ", rbm(v, 10))
    print("Energy : ", e)
    print("Free energy : ", fe)
    print("Reconstruction : ", recon)
    print("Inference : ", inf)
    print("Partition function : ", log_Z)
    print("Likelihood : ", logp)

    e.backward()
    
    print("Units : ", units["v"].data, units["h"].data)
    print("Grad w: ", w.grad)
    print("Grad b: ", b.grad)
    print("Grad c: ", c.grad)


