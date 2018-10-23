"""

param.py

Created by Charles Murphy on 21-08-13.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines the Param.
"""

from copy import copy
import torch
import torch.nn as nn
import numpy as np

MIN_VALUE = -80
MAX_VALUE = 80

__all__ = ['Param', 'Weight', 'Bias', 'Empty_Bias']

class Param(object):
    """docstring for Param"""
    def __init__(self, key, p_kind, use_cuda=False):
        super(Param, self).__init__()
        self.key = key
        self.p_kind = p_kind
        self.use_cuda = use_cuda

    def __repr__(self):
        return "<param.Param>"


    def __str__(self):
        return self.value.__str__()


    def __copy__(self):
        param_copy = Param(self.key,
                           self.p_kind,
                           self.use_cuda)

        param_copy.value = self.value.clone()

        return param_copy


    def phase(self, units):
        raise NotImplementedError('self.phase has not been implemented.')
        return 0


    def mean_term(self, units, key):
        raise NotImplementedError('self.mean_term has not been implemented.')
        return 0


    def energy_term(self, units):
        raise NotImplementedError('self.energy_term has not been implemented.')
        return 0


class Weight(Param):
    """docstring for Weights"""
    def __init__(self, units, init_param, use_cuda):

        key = [units[0].key, units[1].key]
        p_kind = "weight"
        use_cuda = units[0].use_cuda
        self.units = units
        self.unit_size = [units[0].size, units[1].size]

        self.value = torch.zeros(self.unit_size[0],
                                 self.unit_size[1])
        self.init_value(init_param)
        
        if use_cuda:
            self.value = self.value.cuda()

        super(Weight, self).__init__(key, p_kind, use_cuda=use_cuda)


    def __repr__(self):
        return "<param.Param.Weight>"

    def __copy__(self):
        param_copy = Weight(self.units)
        param_copy.value = self.value.clone()

        return param_copy


    def init_value(self, scale=0.01):
        self.value.normal_(0, scale)

        return 0


    def phase(self, units):
        batchsize = units[self.key[0]].batchsize
        unit0 = units[self.key[0]].value
        unit1 = units[self.key[1]].value
        return torch.matmul(unit0.t(), unit1) / batchsize


    def mean_term(self, units, key):
        """
        x : complet input of activation
        key : key of x on which the sum is performed.
        """

        if key == self.key[0]:
            # print(units[self.key[0]].value)
            unit0 = units[self.key[0]].value
            val = torch.matmul(unit0, self.value)
            return torch.clamp(val, MIN_VALUE, MAX_VALUE)
        elif key == self.key[1]:
            unit1 = units[self.key[1]].value
            val = torch.matmul(unit1, self.value.t())
            return torch.clamp(val, MIN_VALUE, MAX_VALUE)

        else:
            raise KeyError("Invalid key in self.mean_term(x, key).\
                            key must be either of these: {}".format(self.key))
            return 0


    def energy_term(self, units):
        unit0 = units[self.key[0]].value
        unit1 = units[self.key[1]].value
        val = -torch.sum(torch.matmul(unit0, self.value) * unit1,dim=1)
        return torch.clamp(val, MIN_VALUE, MAX_VALUE)


    def size(self):
        return self.value.size()


class Bias(Param):
    """docstring for Bias"""
    def __init__(self, unit, init_param, use_cuda):
        p_kind = "bias"
        self.size = unit.size
        self.u_kind = unit.u_kind
        use_cuda = unit.use_cuda
        self.unit = unit

        if unit.s_kind is "bernoulli":
            self.energy_term = self.energy_bernoulli
        elif unit.s_kind is "gaussian":
            self.energy_term = self.energy_gaussian

        self.value = torch.zeros(self.size)
        self.init_value(init_param)

        if use_cuda:
            self.value = self.value.cuda()
        super(Bias, self).__init__(unit.key, p_kind, use_cuda=use_cuda)


    def __repr__(self):
        return "<param.Param.Bias>"

    def __copy__(self):
        param_copy = Bias(self.unit)
        param_copy.value = self.value.clone()

        return param_copy

    def init_value(self, p=None):
        if p is None:
            p = torch.ones(self.size) * 0.5

        self.value = torch.log(p / (1. - p))
        return 0


    def phase(self, units):
        batchsize = units[self.key].batchsize

        unit = units[self.key].value
        return torch.sum(unit, dim=0) / batchsize


    def mean_term(self, units, key):
        batchsize = units[self.key].batchsize

        self.value.resize_(1, self.size)
        val = torch.cat([self.value] * batchsize)
        self.value.resize_(self.size)

        return torch.clamp(val, MIN_VALUE, MAX_VALUE)


    def energy_term(self, units):
        NotImplementedError("self.energy_term() has not been implemented.")
        return 0


    def energy_bernoulli(self, units):
        unit0 = units[self.key].value

        val = -torch.matmul(unit0, self.value)
        return torch.clamp(val, MIN_VALUE, MAX_VALUE)


    def energy_gaussian(self, units):
        unit0 = units[self.key].value

        # val = -torch.matmul(unit0, self.value)
        val = 0.5 * (unit0 - self.value)**2
        return torch.clamp(val, MIN_VALUE, MAX_VALUE)


    def size(self):
        return self.value.size()


if __name__ == '__main__':
    num_unit = 5
    batchsize = 2
    use_cuda = False

    from unit import *
    v_info = Unit_info("v", num_unit, "visible")
    h_info = Unit_info("h", num_unit, "hidden")
    v = Unit("v", v_info, batchsize)
    h = Unit("h", h_info, batchsize)

    x = {"v":v, "h":h}

    weight = Weight((v, h), "xavier-normal")
    weight.value.normal_()
    bias_v = Bias(v)
    empty_bias = Empty_Bias(v)

    bias_v.mean_term(x, "v")

    def show_unit(u):
        if not u.is_super:
            print(u.key)
            print("\n info:", u.unit_info)
            print("\n value:", u.value)
        else:
            print(u.key)
            print("\n info:", [str(i) for i in u.unit_info])
            print("\n value:", u.value)


    def show_params(name, param):

        print(name)
        print("\t Value: ", param.value)
        print("\t Energy term: ", param.energy_term(x))
        print("\t Activation term summed over 'v': ", param.mean_term(x, "v"))
        print("\t Activation term summed over 'h': ", param.mean_term(x, "h"))

    show_unit(x["v"])
    show_unit(x["h"])

    show_params("Weight", weight)
    show_params("Bias", bias_v)
    # show_params("Empty bias", empty_bias)

