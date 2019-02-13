"""

rbm.py

Created by Charles Murphy on 21-07-18.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines the class RBM which generate Restrited Boltzmann
machine. Within this class, the sampling and training 
algorithms are defined.
"""

import torch

from ../..utilities.utilities import sigmoid, random_binary
from .unit import *
from .param import *
from .bm import *

__all__  = ['RBM', 'RBM_BernoulliBernoulli', 'RBM_GaussBernoulli',
            'RBM_GaussGauss']


class RBM(General_Boltzmann_Machine):
    """docstring for RBM"""
    def __init__(self, n_visible, n_hidden, v_s_kind, h_s_kind,
                 config=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        units_info = {'v': Unit_info('v', n_visible, "visible", v_s_kind),
                      'h': Unit_info('h', n_hidden, "hidden", h_s_kind)}
        params_info = ['vh', 'v', 'h']

        super(RBM, self).__init__(units_info, params_info, config)


    # Sampling methods
    def free_energy(self, v):
        
        units = self.init_units({'v': v})
        activation_h = self.params['vh'].mean_term(units, 'v') + \
                       self.params['h'].mean_term(units, 'v')

        freeenergy = self.params['v'].energy_term(units) - \
                   torch.sum(torch.log(1 + torch.exp(activation_h)), 1)
        return freeenergy.mean()

        
    def inference(self, v):
        
        units = self.init_units({'v': v})

        activation_h = self.params['vh'].mean_term(units, 'v') + \
                       self.params['h'].mean_term(units, 'v')
        p = torch.exp(units['h'].log_p(activation_h)).detach()

        prob = {'h': p}
        unit_sample = self.init_units({'v': v,
                                       'h': random_binary(p, self.use_cuda)})
        return prob, unit_sample

    def sampler(self, units, numsteps, given='v', with_pcd=False):

        if with_pcd:
            units = self.mc_units

        v = units["v"].data.clone()
        if given != 'v':
            s_0 = 'v'
            s_1 = 'h'
        else:
            s_0 = 'h'
            s_1 = 'v'

        for i in range(numsteps):
            # sampling hidden unit
            activation_0 = self.params['vh'].mean_term(units, s_1)\
                         + self.params[s_0].mean_term(units, s_1)
            units[s_0].sample(activation_0)

            # sampling visible units
            activation_1 = self.params['vh'].mean_term(units, s_0)\
                         + self.params[s_1].mean_term(units, s_0)
            units[s_1].sample(activation_1)

        self.mc_units = units


        return units


class RBM_BernoulliBernoulli(RBM):
    """docstring for RBM"""
    def __init__(self, n_visible, n_hidden, config=None):

        super(RBM_BernoulliBernoulli, self).__init__(n_visible, n_hidden, 
                                                     "bernoulli", "bernoulli",
                                                     config)

class RBM_GaussBernoulli(RBM):
    """docstring for RBM"""
    def __init__(self, n_visible, n_hidden, config=None):

        super(RBM_GaussBernoulli, self).__init__(n_visible, n_hidden, 
                                                 "gaussian", "bernoulli",
                                                 config)

class RBM_GaussGauss(RBM):
    """docstring for RBM"""
    def __init__(self, n_visible, n_hidden, config=None):

        super(RBM_GaussGauss, self).__init__(n_visible, n_hidden, 
                                             "gaussian", "bernoulli",
                                             config)

