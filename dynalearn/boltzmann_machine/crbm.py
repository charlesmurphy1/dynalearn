"""

crbm.py

Created by Charles Murphy on 21-07-18.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines the class CRBM which generate Conditional Restrited 
Boltzmann machine.
"""

import utilities.utilities as util
from .unit import *
from .param import *
from .bm import *

__all__  = ['CRBM']

class CRBM(General_Boltzmann_Machine):
    """docstring for CRBM"""
    def __init__(self, n_visible, n_conditional, n_hidden, model_config):
        self.n_visible = n_visible
        self.n_conditional = n_conditional
        self.n_hidden = n_hidden

        units_info = {"v": Unit_info("v", n_visible, "visible", 
                                     "bernoulli"),
                      "c": Unit_info("c", n_conditional, "conditional",
                                     "bernoulli"),
                      "h": Unit_info("h", n_hidden, "hidden",
                                     "bernoulli")}
        params_info = [("v", "h"), ("v", "c"), ("h", "c"), "v", "h"]

        super(CRBM, self).__init__(units_info, params_info, model_config)


    def __copy__(self):
        crbm = CRBM(self.n_visible, self.n_conditional, self.n_hidden, 
                    self.model_config)

        for k in crbm.params:
            crbm.params[k].value = self.params[k].value.clone()

        return crbm


    # Sampling methods
    def free_energy(self, v_data):
        units = self.init_units({"v": v_data[0], "c": v_data[1]})

        activation_h = self.params[("v", "h")].mean_term(units, "v") \
                     + self.params[("h", "c")].mean_term(units, "c") \
                     + self.params["h"].mean_term(units, "v")

        energy_terms = self.params["v"].energy_term(units) \
                     + self.params[("v", "c")].energy_term(units)

        val = energy_terms - torch.sum(torch.log(1 + torch.exp(activation_h)), 1)

        return val

        
    def inference(self, v_data):
        units = self.init_units({"v": v_data[0], "c": v_data[1]})

        # print(units["v"])
        activation_h = self.params[("v","h")].mean_term(units, "v") \
                     + self.params[("h", "c")].mean_term(units, "c") \
                     + self.params["h"].mean_term(units, "v")

        return {"v":v_data[0], "c":v_data[1], "h":util.sigmoid(activation_h)}


    def sampler(self, num_steps, units=None, given="v"):

        if units is None:
            raise ValueError("units in self.sampler must be defined pour CRBM.")

        if given != "v":
            s_0 = "v"
            s_1 = "h"
        else:
            s_0 = "h"
            s_1 = "v"

        for i in range(num_steps):
            # sampling hidden unit
            activation_0 = self.params[("v","h")].mean_term(units, s_1) \
                         + self.params[(s_0,"c")].mean_term(units, "c") \
                         + self.params[s_0].mean_term(units, s_1)
            units[s_0].sample(activation_0)

            # sampling visible units
            activation_1 = self.params[("v","h")].mean_term(units, s_0) \
                         + self.params[(s_1,"c")].mean_term(units, "c") \
                         + self.params[s_1].mean_term(units, s_0)
            units[s_1].sample(activation_1)

        self.mc_units = units

        return units



if __name__ == '__main__':
    n_visible = 50
    n_conditonal = 2
    n_hidden = 1
    init_scale = 0.01
    p = None
    batchsize = 1
    use_cuda = False

    crbm = CRBM(n_visible, n_conditonal, n_hidden,
                batchsize,
                init_scale=init_scale,
                p=p,
                use_cuda=use_cuda)

    units = crbm.init_units()
    units = crbm.sampler(units, 10)
    print(crbm.params)

    inference = crbm.inference([units["v"].value, units["c"].value])

    print(inference)
