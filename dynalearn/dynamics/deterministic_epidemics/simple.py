import numpy as np

from .base import (
    DeterministicEpidemics,
    WeightedDeterministicEpidemics,
    MultiplexDeterministicEpidemics,
    WeightedMultiplexDeterministicEpidemics,
)
from dynalearn.config import Config
from dynalearn.nn.models import Propagator

EPSILON = 0.0


class SimpleDSIS(DeterministicEpidemics):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)
        DeterministicEpidemics.__init__(self, config, 2)
        self.infection_prob = config.infection_prob
        self.infection_type = config.infection_type
        self.recovery_prob = config.recovery_prob
        self.propagator = Propagator()

    def update(self, x):
        p = np.zeros((self.num_nodes, self.num_states))
        infection_prob = self.infection(x).squeeze()
        p[:, 0] += self.recovery_prob * x[:, 1]
        p[:, 0] -= infection_prob * x[:, 0] / self.node_degree
        p[:, 1] += infection_prob * x[:, 0] / self.node_degree
        p[:, 1] -= self.recovery_prob * x[:, 1]
        return p

    def infection(self, x):
        I = x[:, 1] * self.population
        if self.infection_type == 1:
            infection = 1 - (1 - self.infection_prob) ** I
        elif self.infection_type == 2:
            infection = 1 - (1 - self.infection_prob / self.population) ** I
        return (
            infection
            + self.propagator(infection, self.edge_index).cpu().detach().numpy()
        )


class WeightedDSIS(SimpleDSIS, WeightedDeterministicEpidemics):
    def __init__(self, config=None, **kwargs):
        SimpleDSIS.__init__(self, config=config, **kwargs)
        WeightedDeterministicEpidemics.__init__(self, config, 2)

    def infection(self, x):
        I = x[:, 1] * self.population
        if self.infection_type == 1:
            infection = 1 - (1 - self.infection_prob) ** I
        elif self.infection_type == 2:
            infection = 1 - (1 - self.infection_prob / self.population) ** I
        return (
            infection
            + self.propagator(infection, self.edge_index, w=self.edge_weight)
            .cpu()
            .detach()
            .numpy()
        )


class MultiplexDSIS(SimpleDSIS, MultiplexDeterministicEpidemics):
    def __init__(self, config=None, **kwargs):
        SimpleDSIS.__init__(self, config=config, **kwargs)
        MultiplexDeterministicEpidemics.__init__(self, config, 2)

    def infection(self, x):
        I = x[:, 1] * self.population
        if self.infection_type == 1:
            infection = 1 - (1 - self.infection_prob) ** I
        elif self.infection_type == 2:
            infection = 1 - (1 - self.infection_prob / self.population) ** I
        return (
            infection
            + self.propagator(infection, self.edge_index["all"]).cpu().detach().numpy()
        )


class WeightedMultiplexDSIS(SimpleDSIS, WeightedMultiplexDeterministicEpidemics):
    def __init__(self, config=None, **kwargs):
        SimpleDSIS.__init__(self, config=config, **kwargs)
        WeightedMultiplexDeterministicEpidemics.__init__(self, config, 2)

    def infection(self, x):
        I = x[:, 1] * self.population
        if self.infection_type == 1:
            infection = 1 - (1 - self.infection_prob) ** I
        elif self.infection_type == 2:
            infection = 1 - (1 - self.infection_prob / self.population) ** I
        return (
            infection
            + self.propagator(
                infection, self.edge_index["all"], w=self.edge_weight["all"]
            )
            .cpu()
            .detach()
            .numpy()
        )


class SimpleDSIR(DeterministicEpidemics):
    def __init__(self, config=None, **kwargs):
        config = config or Config(**kwargs)
        DeterministicEpidemics.__init__(self, config, 3)
        self.infection_prob = config.infection_prob
        self.infection_type = config.infection_type
        self.recovery_prob = config.recovery_prob
        self.propagator = Propagator()

    def update(self, x):
        p = np.zeros((self.num_nodes, self.num_states))
        infection_prob = self.infection(x).squeeze()
        nz_index = self.node_degree > 0.0

        p[nz_index, 0] -= (
            infection_prob[nz_index] * x[nz_index, 0] / self.node_degree[nz_index]
        )
        p[nz_index, 1] += (
            infection_prob[nz_index] * x[nz_index, 0] / self.node_degree[nz_index]
        )

        p[:, 1] -= self.recovery_prob * x[:, 1]
        p[:, 2] += self.recovery_prob * x[:, 1]
        return p

    def infection(self, x):
        I = x[:, 1] * self.population
        if self.infection_type == 1:
            infection = 1 - (1 - self.infection_prob) ** I
        elif self.infection_type == 2:
            infection = 1 - (1 - self.infection_prob / self.population) ** I
        if (
            np.any(infection > 1)
            or np.any(np.isnan(infection))
            or np.any(infection < 0)
        ):
            exit()
        return (
            infection
            + self.propagator(infection, self.edge_index).cpu().detach().numpy()
        )


class WeightedDSIR(SimpleDSIR, WeightedDeterministicEpidemics):
    def __init__(self, config=None, **kwargs):
        SimpleDSIR.__init__(self, config=config, **kwargs)
        WeightedDeterministicEpidemics.__init__(self, config, 3)

    def infection(self, x):
        I = x[:, 1] * self.population
        if self.infection_type == 1:
            infection = 1 - (1 - self.infection_prob) ** I
        elif self.infection_type == 2:
            infection = 1 - (1 - self.infection_prob / self.population) ** I
        return (
            infection
            + self.propagator(infection, self.edge_index, w=self.edge_weight)
            .cpu()
            .detach()
            .numpy()
        )


class MultiplexDSIR(SimpleDSIR, MultiplexDeterministicEpidemics):
    def __init__(self, config=None, **kwargs):
        SimpleDSIR.__init__(self, config=config, **kwargs)
        MultiplexDeterministicEpidemics.__init__(self, config, 3)

    def infection(self, x):
        I = x[:, 1] * self.population
        if self.infection_type == 1:
            infection = 1 - (1 - self.infection_prob) ** I
        elif self.infection_type == 2:
            infection = 1 - (1 - self.infection_prob / self.population) ** I
        return (
            infection
            + self.propagator(infection, self.edge_index["all"]).cpu().detach().numpy()
        )


class WeightedMultiplexDSIR(SimpleDSIR, WeightedMultiplexDeterministicEpidemics):
    def __init__(self, config=None, **kwargs):
        SimpleDSIR.__init__(self, config=config, **kwargs)
        WeightedMultiplexDeterministicEpidemics.__init__(self, config, 3)

    def infection(self, x):
        I = x[:, 1] * self.population
        if self.infection_type == 1:
            infection = 1 - (1 - self.infection_prob) ** I
        elif self.infection_type == 2:
            infection = 1 - (1 - self.infection_prob / self.population) ** I
        return (
            infection
            + self.propagator(
                infection, self.edge_index["all"], w=self.edge_weight["all"]
            )
            .cpu()
            .detach()
            .numpy()
        )


def DSIS(config=None, **kwargs):
    if "is_weighted" in config.__dict__:
        is_weighted = config.is_weighted
    else:
        is_weighted = False

    if "is_multiplex" in config.__dict__:
        is_multiplex = config.is_multiplex
    else:
        is_multiplex = False
    if is_weighted and is_multiplex:
        return WeightedMultiplexDSIS(config=config, **kwargs)
    elif is_weighted and not is_multiplex:
        return WeightedDSIS(config=config, **kwargs)
    elif not is_weighted and is_multiplex:
        return MultiplexDSIS(config=config, **kwargs)
    else:
        return SimpleDSIS(config=config, **kwargs)


def DSIR(config=None, **kwargs):
    if "is_weighted" in config.__dict__:
        is_weighted = config.is_weighted
    else:
        is_weighted = False

    if "is_multiplex" in config.__dict__:
        is_multiplex = config.is_multiplex
    else:
        is_multiplex = False

    if is_weighted and is_multiplex:
        return WeightedMultiplexDSIR(config=config, **kwargs)
    elif is_weighted and not is_multiplex:
        return WeightedDSIR(config=config, **kwargs)
    elif not is_weighted and is_multiplex:
        return MultiplexDSIR(config=config, **kwargs)
    else:
        return SimpleDSIR(config=config, **kwargs)
