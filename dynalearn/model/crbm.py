"""

crbm.py

Created by Charles Murphy on 21-07-18.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines the class CRBM which generate Conditional Restrited 
Boltzmann machine.
"""

import torch

from ..utilities.utilities import sigmoid
from .unit import *
from .param import *
from .bm import *

__all__  = ['CRBM', 'CRBM_BernoulliBernoulli', 'CRBM_GaussBernoulli',
            'CRBM_GaussGauss']

class CRBM(General_Boltzmann_Machine):
    """docstring for CRBM"""
    def __init__(self, n_visible, n_conditional, n_hidden, v_s_kind, 
                 c_s_kind, h_s_kind, model_config=None):
        self.n_visible = n_visible
        self.n_conditional = n_conditional
        self.n_hidden = n_hidden

        units_info = {
                        "v": Unit_info("v", n_visible, "visible",
                                        v_s_kind),
                        "c": Unit_info("c", n_conditional, "conditional",
                                        c_s_kind),
                        "h": Unit_info("h", n_hidden, "hidden",
                                        h_s_kind)
                      }
        params_info = [("v", "h"), ("v", "c"), ("h", "c"), "v", "h"]

        super(CRBM, self).__init__(units_info, params_info, model_config)
    

    def free_energy(self, v):
        
        units = self.init_units({"v": v})
        activation_h = self.params[("v", "h")].mean_term(units, "v") + \
                       self.params[("h", "c")].mean_term(units, "c") + \
                       self.params["h"].mean_term(units, "v")

        freeenergy = self.params["v"].energy_term(units) - \
                     torch.sum(torch.log(1 + torch.exp(activation_h)), 1)
        return freeenergy

        
    def inference(self, v):
        
        units = self.init_units({"v": v})

        activation_h = self.params[("v","h")].mean_term(units, "v")
        activation_h += self.params[("h", "c")].mean_term(units, "c")
        activation_h += self.params["h"].mean_term(units, "v")
        prob = torch.exp(units["h"].log_p(activation_h))
        return {"v": v, "h": prob}

    def conditional_log_p(self, v):
        units = self.init_units({"v": v})
        result = {}

        # hidden unit
        activation_h = self.params[("v", "h")].mean_term(units, "v")
        activation_h = self.params[("h", "c")].mean_term(units, "c")
        activation_h += self.params["h"].mean_term(units, "v")
        result["h"] = units["h"].log_p(activation_h)
        units["h"].sample(activation_h)

        # visible units
        activation_v = self.params[("v", "h")].mean_term(units, "h")
        activation_h = self.params[("v", "c")].mean_term(units, "c")
        activation_v += self.params["v"].mean_term(units, "h")
        result["v"] = units["v"].log_p(activation_v)

        return result


    def sampler(self, units, num_steps, given="v"):

        if units is None:
            units = self.mc_units

        if given != "v":
            s_0 = "v"
            s_1 = "h"
        else:
            s_0 = "h"
            s_1 = "v"

        for i in range(num_steps):
            # sampling hidden unit
            activation_0 = self.params[("v","h")].mean_term(units, s_1) + \
                           self.params[(s_0,"c")].mean_term(units, "c") + \
                           self.params[s_0].mean_term(units, s_1)
            units[s_0].sample(activation_0)

            # sampling visible units
            activation_1 = self.params[("v","h")].mean_term(units, s_0) + \
                           self.params[(s_1,"c")].mean_term(units, "c") + \
                           self.params[s_1].mean_term(units, s_0)
            units[s_1].sample(activation_1)

        self.mc_units = units

        return units


class CRBM_BernoulliBernoulli(CRBM):
    """docstring for RBM"""
    def __init__(self, n_visible, n_conditional, n_hidden, model_config=None):

        super(CRBM_BernoulliBernoulli, self).__init__(n_visible, n_conditional,
                                                      n_hidden, "bernoulli",
                                                      "bernoulli", "bernoulli",
                                                      model_config)

class CRBM_GaussBernoulli(CRBM):
    """docstring for RBM"""
    def __init__(self, n_visible, n_conditional, n_hidden, model_config=None):

        super(CRBM_GaussBernoulli, self).__init__(n_visible, n_conditional,
                                                      n_hidden, "gaussian",
                                                      "gaussian", "bernoulli",
                                                      model_config)

class CRBM_GaussGauss(CRBM):
    """docstring for RBM"""
    def __init__(self, n_visible, n_conditional, n_hidden, model_config=None):

        super(CRBM_GaussGauss, self).__init__(n_visible, n_conditional,
                                                      n_hidden, "gaussian",
                                                      "gaussian", "gaussian",
                                                      model_config)
