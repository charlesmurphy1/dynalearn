"""

unit.py

Created by Charles Murphy on 21-08-13.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines the Unit.
"""

import torch
import numpy as np
import utilities as util


class Unit_info(object):
    """docstring for Unit_info"""
    def __init__(self, key, size, u_kind, s_kind="bernoulli"):
        super().__init__()
        self.key = key
        self.size = size
        self.u_kind = u_kind
        self.s_kind = s_kind


    def __repr__(self):
        return "<unit.Unit_info>"


    def __str__(self):
        return "{0} {1} {2} {3}".format(self.key, self.size,
                                        self.u_kind, self.s_kind)

        


class Unit(object):
    """docstring for Unit"""
    def __init__(self, key, unit_info, batchsize,
                 use_cuda=False):
        super().__init__()

        self.key = key
        self.is_super = util.is_iterable(unit_info)
        self.unit_info = unit_info
        self.batchsize = batchsize
        self.use_cuda = use_cuda

        if self.is_super:
            self.sizes = [u.size for u in unit_info]
            self.keys = [u.key for u in unit_info]
            self.size = sum(self.sizes)
            self.u_kind = "super"
            self.s_kind = next(iter(unit_info)).s_kind
        else:
            self.size = unit_info.size
            self.u_kind = unit_info.u_kind
            self.s_kind = unit_info.s_kind


        if self.s_kind == "bernoulli":
            self.log_p = self.log_p_bernoulli
            self.mean = self.mean_bernoulli
            self.sample = self.sample_bernoulli
        elif self.s_kind == "gaussian":
            self.log_p = self.log_p_gaussian
            self.mean = self.mean_gaussian
            self.sample = self.sample_gaussian
        else:
            raise ValueError("Wrong value of Unit s_kind.")
        # elif self.s_kind == "multinomial":
        #     self.sample = self.sample_multinomial
        
        self.value = torch.zeros([batchsize, self.size])
        self.init_value()

        if self.use_cuda:
            self.value = self.value.cuda()


    def __repr__(self):
        val = "<unit.Unit.single>"
        if self.is_super:
            val = "<unit.Unit.super>"

        return val


    def __str__(self):
        return self.value.__str__()

    def init_value(self):
        self.sample()
        return self.value

    def log_p(self, mean=None):
        raise NotImplementedError('self.probability() has not been implemented.')
        return 0

    def sample(self, mean=None):
        raise NotImplementedError('self.sample() has not been implemented.')
        return 0

    def mean(self, mean=None):
        raise NotImplementedError('self.mean() has not been implemented.')
        return 0

    def log_p_bernoulli(self, mean=None):
        if mean is None:
            mean = torch.zeros(self.value.size())
        if torch.any(self.value > 1):
            raise ValueError("allo gros cave")
        return mean * self.value - torch.log(1 + torch.exp(mean))

    def log_p_gaussian(self, mean=None):
        if mean is None:
            mean = torch.zeros(self.value.size())
        return -(self.value - mean)**2 - 0.5 * np.log(2 * np.pi)


    def mean_bernoulli(self, mean=None):
        if mean is None:
            mean = torch.zeros(self.value.size())

        return util.sigmoid(mean)


    def mean_gaussian(self, mean=None):
        if mean is None:
            mean = torch.zeros(self.value.size())

        return mean

    def sample_bernoulli(self, mean=None):
        p = self.mean(mean)
        self.value = torch.bernoulli(p)

        return self.value


    def sample_gaussian(self, mean=None):
        mean = self.mean(mean)
        self.value = torch.normal(mean)

        return self.value
    

    def sample_multinomial(self, mean=None):
        if mean is None:
            mean = torch.zeros(self.value.size())
        # p = torch.exp(mean)
        # torch.multinomial(p, 1, True)
        raise NotImplementedError('self.sample_multinomial() has not been implemented.')
        return self.value



if __name__ == '__main__':
    u_i1 = Unit_info("v1", 2, "visible", "bernoulli")
    u_i2 = Unit_info("v2", 2, "visible", "gaussian")
    batchsize = 5


    u1 = Unit("v1", u_i1, batchsize)
    # u1.value = torch.ones(u1.batchsize, u1.size)
    u2 = Unit("v2", u_i2, batchsize)
    u3 = Unit("v3", [u_i1, u_i2], batchsize)
    u3.value = torch.cat([u1.value, u2.value], 1)

    def show_unit(u):
        if not u.is_super:
            print(u.key)
            print("\n info:", u.unit_info)
            print("\n value:", u.value)
        else:
            print(u.key)
            print("\n info:", [str(i) for i in u.unit_info])
            print("\n value:", u.value)

    show_unit(u1)
    show_unit(u2)
    show_unit(u3)



