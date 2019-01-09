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

__all__ = ['Param', 'Weight', 'Bias']
PARAM_MIN = -80
PARAM_MAX = 80

class Param(nn.Module):
    """docstring for Param"""
    def __init__(self, key, p_kind, use_cuda=False):
        super(Param, self).__init__()
        self.key = key
        self.p_kind = p_kind
        self.use_cuda = use_cuda
        self.param = nn.Parameter()

    def __repr__(self):
        return self.param.__str__()


    def __str__(self):
        return self.param.__str__()


    def __copy__(self):
        raise NotImplementedError('self.__copy__ has not been implemented.')

    # def phase(self, units):
    #     raise NotImplementedError('self.phase has not been implemented.')
    #     return 0


    def mean_term(self, units, key):
        raise NotImplementedError('self.mean_term has not been implemented.')


    def energy_term(self, units):
        raise NotImplementedError('self.energy_term has not been implemented.')


class Weight(Param):
    """docstring for Weights"""
    def __init__(self, units, init_param=0.01, use_cuda=False):

        key = [units[0].key, units[1].key]
        p_kind = "weight"
        self.units = units
        self.unit_size = [units[0].size, units[1].size]

        super(Weight, self).__init__(key, p_kind, use_cuda=use_cuda)

        self.param = nn.Parameter(
                                    torch.zeros(self.unit_size[0], 
                                                self.unit_size[1])
                                 )

        self.init_value(init_param)
        
        if use_cuda:
            self.param.data = self.param.data.cuda()



    def __copy__(self):
        copy = Weight(self.units, use_cuda=self.use_cuda)
        copy.param.data = self.param.data.clone()

        return copy


    def init_value(self, scale=0.01):
        self.param.data.normal_(0., scale)



    def mean_term(self, units, key):
        """
        x : complet input of activation
        key : key of x on which the sum is performed.
        """
        
        if key == self.key[0]:
            unit0 = units[self.key[0]].data
            val = torch.matmul(unit0, self.param.data)
            return val
        elif key == self.key[1]:
            unit1 = units[self.key[1]].data
            val = torch.matmul(unit1, self.param.data.t())
            return val

        else:
            raise KeyError("Invalid key in self.mean_term(x, key).\
                            key must be either of these: {}".format(self.key))
            return 0


    def energy_term(self, units):
        unit0 = units[self.key[0]].data
        unit1 = units[self.key[1]].data
        val = -torch.sum(torch.matmul(unit0, self.param) * unit1,dim=1)

        return val.mean()


    def size(self):
        return self.param.size()


class Bias(Param):
    """docstring for Bias"""
    def __init__(self, unit, init_param=None, use_cuda=False):
        p_kind = "bias"
        self.size = unit.size
        self.u_kind = unit.u_kind
        self.unit = unit

        super(Bias, self).__init__(unit.key, p_kind, use_cuda=use_cuda)
        
        if unit.s_kind is "bernoulli":
            self.energy_term = self.__energy_bernoulli
        elif unit.s_kind is "gaussian":
            self.energy_term = self.__energy_gaussian

        self.param = nn.Parameter(torch.zeros(self.size))
        self.init_value(init_param)

        if use_cuda:
            self.param.data = self.param.data.cuda()


    def __copy__(self):
        copy = Bias(self.unit)
        copy.data = self.param.data.clone()

        return copy


    def forward():
        return None

    def init_value(self, p=None):
        if p is None:
            p = torch.ones(self.size) * 0.5
        elif type(p) is float:
            p = torch.ones(self.size) * p
        elif len(p) != self.size:
            n = len(p)
            raise ValueError(f"Expected p size to be {self.size},\
                               got instead {n}.")
        self.param.data = torch.log(p / (1. - p))
        self.param.data[self.param.data < PARAM_MIN] = PARAM_MIN
        self.param.data[self.param.data > PARAM_MAX] = PARAM_MAX

    def mean_term(self, units, key):
        batchsize = units[self.key].batchsize

        val = self.param.repeat([batchsize, 1])

        return val


    def energy_term(self, units):
        NotImplementedError("self.energy_term() has not been implemented.")
        return 0


    def __energy_bernoulli(self, units):
        unit0 = units[self.key].data

        val = -torch.matmul(unit0, self.param)
        return val.mean()


    def __energy_gaussian(self, units):
        unit0 = units[self.key].data

        # val = -torch.matmul(unit0, self.param)
        val = (0.5 * (unit0 - self.param)**2)
        return val.mean()


    def size(self):
        return self.param.size()

