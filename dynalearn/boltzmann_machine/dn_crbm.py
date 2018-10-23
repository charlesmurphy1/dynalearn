"""

crbm.py

Created by Charles Murphy on 21-07-18.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines the class DynNet_CRBM which generate Conditional Restrited 
Boltzmann machine for dynamical networks.
"""

import bm
import utilities as util
from units import *

class DynNet_CRBM(bm.General_Boltzmann_Machine):
    """docstring for DynNet_CRBM"""
    def __init__(self, n_dim, n_neighbors_max, n_hidden, memory, batchsize,
                 init_mode="xavier-normal", use_cuda=False):
        self.n_dim = n_dim
        self.n_neighbors_max = n_neighbors_max
        self.n_hidden = n_hidden
        self.memory = memory


        # passed states
        passed_state_units_info = []
        for i in range(self.memory):
            u_i_passed = Unit_info("p{0}".format(i), n_dim, "conditional")
            passed_state_units_info.append(u_i_passed)

        # neighbors passed states
        neighbors_passed_state_units_info = []
        for n in range(self.n_neighbors_max):
            for i in range(self.memory):
                u_i_neighbor = Unit_info("n{0}p{1}".format(n, i),
                                         n_dim, "conditional")
                neighbors_passed_state_units_info.append(u_i_neighbor)

        units_info = {
                    # Visible units
                    "x": Unit_info("v", n_dim, "visible"),
                    # Hidden units
                    "h": Unit_info("h", n_hidden, "hidden"),  
                    # Passed conditional super_units
                    "p": passed_state_units_info,             
                    # Neighbors passed conditional super_units
                    "np": neighbors_passed_state_units_info
                    }

        # Define weights
        weights_info = [("x", "h"),
                        ("x", "p"), ("x", "np"),
                        ("h", "p"), ("h", "np")]

        # Define parent class
        super().__init__(units_info, weights_info, batchsize,
                         init_mode=init_mode, use_cuda=use_cuda)


    # Sampling methods
    def inference_probability(self, x, p, np):
        units = self.init_units()
        # print(p.value, units["p"].value.size())
        units["x"].value = x
        units["p"].value = p
        units["np"].value = np

        act_h = self.params[("x","h")].activation_term(units, "x")
        act_h += self.params[("h", "p")].activation_term(units, "p")
        act_h += self.params[("h", "np")].activation_term(units, "np")
        act_h += self.params["h"].activation_term(units, "x")

        return util.sigmoid(act_h)


    def sample_inference(self, x, p, np):
        h = self.inference_probability(x, p, np)

        units = self.init_units()
        units["x"].value = x
        units["p"].value = p
        units["np"].value = np
        units["h"].value = h

        return units


    def sampler(self, num_steps, units):

        if units is None:
            raise ValueError("in self.sampler, units must be not None.\
                             Use sample from training dataset (CD).")

        for i in range(num_steps):
            # sampling hidden units
            act_h = self.params[("x","h")].activation_term(units, "x")
            act_h += self.params[("h", "p")].activation_term(units, "p")
            act_h += self.params[("h", "np")].activation_term(units, "np")
            # print(act_h.size(), self.params["h"].activation_term(units, "x").size())
            act_h += self.params["h"].activation_term(units, "x")

            prob_h = util.sigmoid(act_h)
            units["h"].value.bernoulli_(prob_h)

            # sampling visible units
            act_x = self.params[("x","h")].activation_term(units, "h")
            act_x += self.params[("x", "p")].activation_term(units, "p")
            act_x += self.params[("x", "np")].activation_term(units, "np")
            act_x += self.params["x"].activation_term(units, None)

            prob_x = util.sigmoid(act_x)
            units["x"].value.bernoulli_(prob_x)

        return units



if __name__ == '__main__':
    
    n_dim = 2
    n_neighbors_max = 4
    n_hidden = 10
    memory = 2
    batchsize = 3
    init_mode="xavier-normal"
    use_cuda=False

    dn_crbm = DynNet_CRBM(n_dim, n_neighbors_max, n_hidden, memory, batchsize,
                          init_mode=init_mode, use_cuda=use_cuda)

    units = dn_crbm.init_units()
    units = dn_crbm.sampler(10, units)
    inference = dn_crbm.sample_inference(units["x"].value,
                                         units["p"].value,
                                         units["np"].value)


    def show_unit_values(name, u):
        print(name)
        print("\t x value: ", u["x"].value)
        print("\t h value: ", u["h"].value)
        print("\t p value: ", u["p"].value)
        print("\t np value: ", u["np"].value)


    show_unit_values("Units", units)
    show_unit_values("Inference", inference)

