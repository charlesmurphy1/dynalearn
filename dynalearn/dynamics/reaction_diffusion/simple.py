import numpy as np

from .base import (
    ReactionDiffusion,
    WeightedReactionDiffusion,
    MultiplexReactionDiffusion,
    WeightedMultiplexReactionDiffusion,
)
from dynalearn.config import Config

EPSILON = 0.0


class SimpleRDSIS(ReactionDiffusion):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        ReactionDiffusion.__init__(self, config, 2)
        self.infection_prob = config.infection_prob
        self.infection_type = config.infection_type
        self.recovery_prob = config.recovery_prob
        self.diffusion_prob = np.array(
            [config.diffusion_susceptible, config.diffusion_infected]
        )

    def reaction(self, x):
        p = np.zeros((self.num_nodes, self.num_states, self.num_states))
        if self.infection_type == 1:
            p[:, 0, 0] = (1 - self.infection_prob) ** x[:, 1]
            p[:, 0, 1] = 1 - (1 - self.infection_prob) ** x[:, 1]
        elif self.infection_type == 2:
            n = x.sum(-1)
            p[:, 0, 0] = (1 - self.infection_prob / n) ** x[:, 1]
            p[:, 0, 1] = 1 - (1 - self.infection_prob / n) ** x[:, 1]
        p[:, 1, 0] = self.recovery_prob
        p[:, 1, 1] = 1 - self.recovery_prob
        return p

    def diffusion(self, x):
        p = {}
        for i, (u, v) in enumerate(self.edge_index.T):
            k = self.node_degree[int(v)]
            p[int(u), int(v)] = self.diffusion_prob / k
        return p


class WeightedRDSIS(SimpleRDSIS, WeightedReactionDiffusion):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        WeightedReactionDiffusion.__init__(self, config, 2)
        SimpleRDSIS.__init__(self, config=config, **kwargs)

    def diffusion(self, x):
        p = {}
        for i, (u, v) in enumerate(self.edge_index.T):
            s = self.node_strength[int(v)]
            w = self.edge_weight[i]
            diff_prob = w / x[int(v)]
            # p[int(u), int(v)] = diff_prob * w / s
            if np.all(s > 0):
                p[int(u), int(v)] = self.diffusion_prob * w / s
            else:
                p[int(u), int(v)] = 0

        return p


class SimpleRDSIR(ReactionDiffusion):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        ReactionDiffusion.__init__(self, config, 3)
        self.infection_prob = config.infection_prob
        self.infection_type = config.infection_type
        self.recovery_prob = config.recovery_prob
        self.diffusion_prob = np.array(
            [
                config.diffusion_susceptible,
                config.diffusion_infected,
                config.diffusion_recovered,
            ]
        )

    def reaction(self, x):
        p = np.zeros((self.num_nodes, self.num_states, self.num_states))
        if self.infection_type == 1:
            p[:, 0, 0] = (1 - self.infection_prob) ** x[:, 1]
            p[:, 0, 1] = 1 - (1 - self.infection_prob) ** x[:, 1]
        elif self.infection_type == 2:
            n = x.sum(-1)
            index = np.where(np.logical_and(n >= 1, x[:, 1] >= 1))[0]
            p[:, 0, 0] = 1
            p[:, 0, 1] = 0
            p[index, 0, 0] = (1 - self.infection_prob / n[index]) ** x[index, 1]
            p[index, 0, 1] = 1 - (1 - self.infection_prob / n[index]) ** x[index, 1]
        p[:, 0, 2] = EPSILON
        p[:, 1, 0] = EPSILON
        p[:, 1, 1] = 1 - self.recovery_prob - EPSILON / 2.0
        p[:, 1, 2] = self.recovery_prob - EPSILON / 2.0
        p[:, 2, 0] = EPSILON / 2.0
        p[:, 2, 1] = EPSILON / 2.0
        p[:, 2, 2] = 1 - EPSILON

        return p

    def diffusion(self, x):
        p = {}
        for i, (u, v) in enumerate(self.edge_index.T):
            k = self.node_degree[int(v)]
            p[int(u), int(v)] = self.diffusion_prob / k
        return p

    def is_dead(self, x):
        if np.all(x[:, 1] == 0):
            return True
        else:
            return False


class WeightedRDSIR(SimpleRDSIR, WeightedReactionDiffusion):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        WeightedReactionDiffusion.__init__(self, config, 3)
        SimpleRDSIR.__init__(self, config=config, **kwargs)

    def diffusion(self, x):
        p = {}
        for i, (u, v) in enumerate(self.edge_index.T):
            s = self.node_strength[int(v)]
            w = self.edge_weight[i]
            if np.all(s > 0):
                p[int(u), int(v)] = self.diffusion_prob * w / s
            else:
                p[int(u), int(v)] = 0
        return p


def RDSIS(config=None, **kwargs):
    config = config or Config(**kwargs)
    if "is_weighted" in config.__dict__:
        is_weighted = config.is_weighted
    else:
        is_weighted = False

    if is_weighted:
        return WeightedRDSIS(config=config, **kwargs)
    else:
        return SimpleRDSIS(config=config, **kwargs)


def RDSIR(config=None, **kwargs):
    config = config or Config(**kwargs)
    if "is_weighted" in config.__dict__:
        is_weighted = config.is_weighted
    else:
        is_weighted = False

    if is_weighted:
        return WeightedRDSIR(config=config, **kwargs)
    else:
        return SimpleRDSIR(config=config, **kwargs)
