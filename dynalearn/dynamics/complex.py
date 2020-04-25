from dynalearn.dynamics.epidemics import SingleEpidemics
from dynalearn.dynamics.activation import constant, threshold, nonlinear, sine, planck
import networkx as nx
import numpy as np


class ComplexSIS(SingleEpidemics):
    def __init__(self, config, activation, deactivation):
        super(ComplexSIS, self).__init__(config, 2)
        self.activation = activation
        self.deactivation = deactivation

    def predict(self, x):
        ltp = np.zeros((x.shape[0], self.num_states))
        l = self.neighbors_state(x)
        p = self.activation(l)
        q = self.deactivation(l)
        ltp[x == 0, 0] = 1 - p[x == 0]
        ltp[x == 0, 1] = p[x == 0]
        ltp[x == 1, 0] = q[x == 1]
        ltp[x == 1, 1] = 1 - q[x == 1]
        return ltp


class ComplexSIR(SingleEpidemics):
    def __init__(self, config, activation, deactivation):
        super(ComplexSIR, self).__init__(config, 2)
        self.activation = activation
        self.deactivation = deactivation

    def predict(self, x):
        ltp = np.zeros((x.shape[0], self.num_states))
        l = self.neighbors_state(x)
        p = self.activation(l)
        q = self.deactivation(l)
        ltp[x == 0, 0] = 1 - p[x == 0]
        ltp[x == 0, 1] = p[x == 0]
        ltp[x == 0, 2] = 0
        ltp[x == 1, 0] = 0
        ltp[x == 1, 1] = 1 - q[x == 1]
        ltp[x == 1, 2] = q[x == 1]
        ltp[x == 2, 0] = 0
        ltp[x == 2, 1] = 0
        ltp[x == 2, 2] = 1
        return ltp


class ThresholdSIS(ComplexSIS):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs

        activation = lambda l: threshold(l[1], l.sum(0), config.threshold, config.slope)
        deactivation = lambda l: constant(l[0], config.recovery)

        super(ThresholdSIS, self).__init__(config, activation, deactivation)


class ThresholdSIR(ComplexSIR):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs

        activation = lambda l: threshold(l[1], l.sum(0), config.threshold, config.slope)
        deactivation = lambda l: constant(l[0], config.recovery)

        super(ThresholdSIR, self).__init__(config, activation, deactivation)


class NonLinearSIS(ComplexSIS):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs

        activation = lambda l: nonlinear(l[1], config.infection, config.exponent)
        deactivation = lambda l: constant(l[0], config.recovery)

        super(NonLinearSIS, self).__init__(config, activation, deactivation)


class NonLinearSIR(ComplexSIR):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs

        activation = lambda l: nonlinear(l[1], config.infection, config.exponent)
        deactivation = lambda l: constant(l[0], config.recovery)

        super(NonLinearSIR, self).__init__(config, activation, deactivation)


class SineSIS(ComplexSIS):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs

        activation = lambda l: sine(
            l[1], config.infection, config.amplitude, config.period
        )
        deactivation = lambda l: constant(l[0], config.recovery)

        super(SineSIS, self).__init__(config, activation, deactivation)


class SineSIR(ComplexSIR):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs

        activation = lambda l: sine(
            l[1], config.infection, config.amplitude, config.period
        )
        deactivation = lambda l: constant(l[0], config.recovery)

        super(SineSIR, self).__init__(config, activation, deactivation)


class PlanckSIS(ComplexSIS):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs

        activation = lambda l: planck(l[1], config.temperature)
        deactivation = lambda l: constant(l[0], config.recovery)

        super(PlanckSIS, self).__init__(config, activation, deactivation)


class PlanckSIR(ComplexSIR):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs

        activation = lambda l: planck(l[1], config.temperature)
        deactivation = lambda l: constant(l[0], config.recovery)

        super(PlanckSIR, self).__init__(config, activation, deactivation)
