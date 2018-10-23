"""

unit.py

Created by Charles Murphy on 21-08-13.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines the Unit.
"""

import torch
import numpy as np
import utilities.utilities as util


__all__ = ['Unit_info', 'Unit']


class Unit_info(object):
    """
    Class containing unit information.

    **Parameters**
    key
        Key to access unit.

    size : Integer
        Size of the unit.

    u_kind : String
        Kind of unit: (usually "visible" or "hidden")

    s_kind : String (either = ["bernoulli", "gaussian"])
        Kind of sampling to perform on the unit.

    """
    def __init__(self, key, size, u_kind, s_kind="bernoulli"):
        super().__init__()
        self.key = key
        self.size = size
        self.u_kind = u_kind

        if (s_kind is not "bernoulli") or (s_kind is not "gaussian"):
            raise ValueError("s_kind in Unit_info.__init__() must be:\
                            ['bernoulli', 'gaussian'].")
        else:
            self.s_kind = s_kind


    def __repr__(self):
        return "<unit.Unit_info>"


    def __str__(self):
        dict_info = {"key": self.key,
                     "size": self.size,
                     "u_kind": self.u_kind,
                     "s_kind": self.s_kind}
        return dict_info.__str__()      


class Unit(object):
    """
    Class defining a unit (unit group).

    **Parameters**
    key
        Key to access unit.

    unit_info : unit.Unit_info
        Information of the unit.

    batchsize : Integer
        Size of the minibatch.

    use_cuda : Bool (default = ``False``)
        Using cuda for parallel GPU processing.

    ..warning::
        If ``True``, Nvidia GPU must be available.

    """
    def __init__(self, key, unit_info, batchsize, use_cuda=False):
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
        """
        Initializes value of unit for random values.

        """
        self.sample()
        return self.value

    def log_p(self, activation=None):
        """
        Computes the log-probability.
        
        **Parameters**
        activation : torch.Tensor : (default = ``None``)
            Activation of given unit values.
        """
        raise NotImplementedError('self.probability() has not been implemented.')
        return 0

    def sample(self, activation=None):
        """
        Samples the value of the unit.
        
        **Parameters**
        activation : torch.Tensor : (default = ``None``)
            Activation of given unit values.
        """
        raise NotImplementedError('self.sample() has not been implemented.')
        return 0

    def activation(self, mean=None):
        """
        Computes the mean (or activation) of given units.
        
        **Parameters**
        mean : torch.Tensor : (default = ``None``)
            Mean of given unit values.
        """
        raise NotImplementedError('self.mean() has not been implemented.')
        return 0

    def log_p_bernoulli(self, mean=None):
        """
        Computes the log-probability for bernoulli units.
        
        **Parameters**
        mean : torch.Tensor : (default = ``None``)
            Mean of given unit values.
        """
        if mean is None:
            mean = torch.zeros(self.value.size())
        if torch.any(self.value > 1):
            raise ValueError("allo gros cave")
        return mean * self.value - torch.log(1 + torch.exp(mean))

    def log_p_gaussian(self, mean=None):
        """
        Computes the log-probability for gaussian units.
        
        **Parameters**
        mean : torch.Tensor : (default = ``None``)
            Mean of given unit values.
        """
        if mean is None:
            mean = torch.zeros(self.value.size())
        return -(self.value - mean)**2 - 0.5 * np.log(2 * np.pi)


    def activation_bernoulli(self, mean=None):
        if mean is None:
            mean = torch.zeros(self.value.size())

        return util.sigmoid(mean)


    def activation_gaussian(self, mean=None):
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



