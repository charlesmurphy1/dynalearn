"""

unit.py

Created by Charles Murphy on 21-08-13.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines the Unit.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from ../..utilities.utilities import is_iterable, sigmoid


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

        if (s_kind != "bernoulli") and (s_kind != "gaussian"):
            raise ValueError("s_kind in Unit_info.__init__() must be:\
                            ['bernoulli', 'gaussian'].")
        else:
            self.s_kind = s_kind


    def __str__(self):
        dict_info = {"key": self.key,
                     "size": self.size,
                     "u_kind": self.u_kind,
                     "s_kind": self.s_kind}
        return dict_info.__str__()      


class Unit(object):
    """
    Class for defining a unit (unit group).

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
        self.is_super = is_iterable(unit_info)
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
            self.log_p = self.__log_p_bernoulli
            self.activation = self.__activation_bernoulli
            self.sample = self.__sample_bernoulli
        elif self.s_kind == "gaussian":
            self.log_p = self.__log_p_gaussian
            self.activation = self.__activation_gaussian
            self.sample = self.__sample_gaussian
        else:
            raise ValueError("Wrong value of Unit s_kind.")
        
        self.data = torch.zeros(batchsize, self.size)
        self.init_value()

        if self.use_cuda:
            self.data = self.data.cuda()


    def __str__(self):
        return self.data.__str__()



    def init_value(self):
        """
        Initializes value of unit for random values.

        """
        return self.sample()

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
        raise NotImplementedError('self.activation() has not been implemented.')
        return 0

    def __log_p_bernoulli(self, mean=None):
        """
        Computes the log-probability for Bernoulli units.
        
        **Parameters**
        mean : torch.Tensor : (default = ``None``)
            Mean of given unit values.
        """
        if mean is None:
            mean = torch.zeros(self.data.size())


        return torch.log(sigmoid(mean))

    def __log_p_gaussian(self, mean=None):
        """
        Computes the log-probability for Gauss units.
        
        **Parameters**
        mean : torch.Tensor : (default = ``None``)
            Mean of given unit values.
        """
        if mean is None:
            mean = torch.zeros(self.data.size())
        return -(self.data - mean)**2 - 0.5 * np.log(2 * np.pi)


    def __activation_bernoulli(self, mean=None):
        if mean is None:
            mean = torch.zeros(self.data.size())

        return sigmoid(mean)


    def __activation_gaussian(self, mean=None):
        if mean is None:
            mean = torch.zeros(self.data.size())

        return mean

    def __sample_bernoulli(self, mean=None):
        p = self.activation(mean).detach()
        self.data.bernoulli_(p)

        return self.data


    def __sample_gaussian(self, mean=None):
        mean = self.activation(mean).detach()
        self.data.normal_(mean)

        return self.data



