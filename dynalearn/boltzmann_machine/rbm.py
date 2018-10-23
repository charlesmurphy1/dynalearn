"""

rbm.py

Created by Charles Murphy on 21-07-18.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines the class RBM which generate Restrited Boltzmann
machine. Within this class, the sampling and training 
algorithms are defined.
"""

import utilities.utilities as util
from .unit import *
from .param import *
from .bm import *

__all__  = ['RBM_Bernoulli', 'RBM_Gauss']


class RBM(General_Boltzmann_Machine):
    """docstring for RBM"""
    def __init__(self, n_visible, n_hidden, model_config):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.v_kind = v_kind

        units_info = {"v": Unit_info("v", n_visible, "visible", "bernoulli"),
                      "h": Unit_info("h", n_hidden, "hidden", "bernoulli")}
        params_info = [("v", "h"), "v", "h"]

        super(RBM, self).__init__(units_info, params_info, model_config)


    def __copy__(self):
        scale = 0.01
        p = None
        rbm = RBM(self.n_visible, self.n_hidden, self.v_kind,
                 scale, p, self.use_cuda)

        for k in rbm.params:
            rbm.params[k].value = self.params[k].value.clone()

        return rbm


    # Sampling methods
    def free_energy(self, v):
        units = self.init_units({"v": v})
        activation_h = self.params[("v", "h")].mean_term(units, "v") + \
                       self.params["h"].mean_term(units, "v")

        freeenergy = self.params["v"].energy_term(units) - \
                   torch.sum(torch.log(1 + torch.exp(activation_h)), 1)
        return freeenergy

        
    def inference(self, v):
        units = self.init_units({"v": v})

        # print(units["v"])
        activation_h = self.params[("v","h")].mean_term(units, "v")
        activation_h += self.params["h"].mean_term(units, "v")

        return self.init_units({"v": v, "h":util.sigmoid(activation_h)})

    def conditional_log_p(self, v):
        
        units = self.init_units({"v": v})
        result = {}

        # hidden unit
        activation_h = self.params["h"].mean_term(units, "v")
        result["h"] = units["h"].log_p(activation_h)
        units["h"].sample(activation_h)

        # visible units
        activation_v = self.params["v"].mean_term(units, "h")
        result["v"] = units["v"].log_p(activation_v)

        return result


    def sampler(self, num_steps, units=None, given="v"):

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
            activation_0 = self.params[("v","h")].mean_term(units, s_1)\
                         + self.params[s_0].mean_term(units, s_1)
            # prob_0 = util.sigmoid(activation_0)
            units[s_0].sample(activation_0)

            # sampling visible units
            activation_1 = self.params[("v","h")].mean_term(units, s_0)\
                         + self.params[s_1].mean_term(units, s_0)
            units[s_1].sample(activation_1)

        self.mc_units = units

        return units
        






if __name__ == '__main__':
    
    n_visible = 5
    n_hidden = 1
    init_scale = 0.01
    # v_kind = "bernoulli"
    v_kind = "gaussian"
    p = None
    batchsize = 1
    use_cuda = False

    rbm = RBM(n_visible, n_hidden,
              v_kind=v_kind,
              init_scale=init_scale,
              p=p,
              use_cuda=use_cuda)

    
    units = rbm.init_units(value_dict=None, batchsize=5)
    print(units["v"], units["h"], torch.mean(units["v"].value))
    print(rbm.conditional_log_p(units["v"].value))

    # rbm_no_w = RBM_no_Weight(n_visible, n_hidden,
    #                       v_kind=v_kind,
    #                       p=p,
    #                       use_cuda=use_cuda)

    # units = rbm_no_w.sampler(None, 10, "v")
    # print(units["v"], units["h"], torch.mean(units["v"].value))


    # inference = rbm.inference(units["v"].value)