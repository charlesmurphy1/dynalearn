import numpy as np

from dynalearn.dynamics.metapopulation import (
    MetaPop,
    WeightedMetaPop,
    MultiplexMetaPop,
    WeightedMultiplexMetaPop,
)
from dynalearn.config import Config

EPSILON = 0.0


class SimpleMetaSIS(MetaPop):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        MetaPop.__init__(self, config, 2)
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


class WeightedMetaSIS(SimpleMetaSIS, WeightedMetaPop):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        WeightedMetaPop.__init__(self, config, 2)
        SimpleMetaSIS.__init__(self, config=config, **kwargs)

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


class MultiplexMetaSIS(SimpleMetaSIS, MultiplexMetaPop):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        MultiplexMetaPop.__init__(self, config, 2)
        SimpleMetaSIS.__init__(self, config=config, **kwargs)

    def diffusion(self, x):
        p = {}
        for i, (u, v) in enumerate(self.edge_index.T):
            k = self.node_degree["all"][int(v)]
            p[int(u), int(v)] = self.diffusion_prob / k
        return p


class WeightedMultiplexMetaSIS(SimpleMetaSIS, WeightedMultiplexMetaPop):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        WeightedMultiplexMetaPop.__init__(self, config, 2)
        SimpleMetaSIS.__init__(self, config=config, **kwargs)

    def diffusion(self, x):
        p = {}
        for i, (u, v) in enumerate(self.edge_index.T):
            s = self.node_strength["all"][int(v)]
            w = self.edge_weight["all"][i]
            diff_prob = w / x[int(v)]
            # p[int(u), int(v)] = diff_prob * w / s
            if np.all(s > 0):
                p[int(u), int(v)] = self.diffusion_prob * w / s
            else:
                p[int(u), int(v)] = 0
        return p


class SimpleMetaSIR(MetaPop):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        MetaPop.__init__(self, config, 3)
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


class WeightedMetaSIR(SimpleMetaSIR, WeightedMetaPop):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        WeightedMetaPop.__init__(self, config, 3)
        SimpleMetaSIR.__init__(self, config=config, **kwargs)

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


class MultiplexMetaSIR(SimpleMetaSIR, MultiplexMetaPop):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        MultiplexMetaPop.__init__(self, config, 3)
        SimpleMetaSIR.__init__(self, config=config, **kwargs)

    def diffusion(self, x):
        p = {}
        for i, (u, v) in enumerate(self.edge_index["all"].T):
            k = self.node_degree["all"][int(v)]
            p[int(u), int(v)] = self.diffusion_prob / k
        return p


class WeightedMultiplexMetaSIR(SimpleMetaSIR, WeightedMultiplexMetaPop):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        WeightedMultiplexMetaPop.__init__(self, config, 3)
        SimpleMetaSIR.__init__(self, config=config, **kwargs)

    def diffusion(self, x):
        p = {}
        for i, (u, v) in enumerate(self.edge_index["all"].T):
            s = self.node_strength["all"][int(v)]
            w = self.edge_weight["all"][i]
            if s > 0:
                p[int(u), int(v)] = self.diffusion_prob * w / s
            else:
                p[int(u), int(v)] = 0
        return p


def MetaSIS(config=None, **kwargs):
    if config is None:
        config = Config()
        config.__dict__ = kwagrs
    if "is_weighted" in config.__dict__:
        is_weighted = config.is_weighted
    else:
        is_weighted = False

    if "is_multiplex" in config.__dict__:
        is_multiplex = config.is_multiplex
    else:
        is_multiplex = False
    if is_weighted and is_multiplex:
        return WeightedMultiplexMetaSIS(config=config, **kwargs)
    elif is_weighted and not is_multiplex:
        return WeightedMetaSIS(config=config, **kwargs)
    elif not is_weighted and is_multiplex:
        return MultiplexMetaSIS(config=config, **kwargs)
    else:
        return SimpleMetaSIS(config=config, **kwargs)


def MetaSIR(config=None, **kwargs):
    if config is None:
        config = Config()
        config.__dict__ = kwagrs
    if "is_weighted" in config.__dict__:
        is_weighted = config.is_weighted
    else:
        is_weighted = False

    if "is_multiplex" in config.__dict__:
        is_multiplex = config.is_multiplex
    else:
        is_multiplex = False

    if is_weighted and is_multiplex:
        return WeightedMultiplexMetaSIR(config=config, **kwargs)
    elif is_weighted and not is_multiplex:
        return WeightedMetaSIR(config=config, **kwargs)
    elif not is_weighted and is_multiplex:
        return MultiplexMetaSIR(config=config, **kwargs)
    else:
        return SimpleMetaSIR(config=config, **kwargs)
