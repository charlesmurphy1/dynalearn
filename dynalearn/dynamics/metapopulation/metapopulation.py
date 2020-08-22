import numpy as np

from dynalearn.dynamics.dynamics import (
    Dynamics,
    WeightedDynamics,
    MultiplexDynamics,
    WeightedMultiplexDynamics,
)
from abc import abstractmethod
from itertools import product


class MetaPop(Dynamics):
    def __init__(self, config, num_states):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        if "state_dist" in config.__dict__:
            self.state_dist = config.state_dist
        else:
            self.state_dist = -1
        if "density" in config.__dict__:
            self.density = config.density
        else:
            self.density = -1
        Dynamics.__init__(self, config, num_states)

    @abstractmethod
    def reaction(self, x):
        raise NotImplemented()

    @abstractmethod
    def diffusion(self, x):
        raise NotImplemented()

    def initial_state(self, state_dist=None, density=None):
        if state_dist is None:
            state_dist = self.state_dist

        if density is None:
            density = self.density
        if not isinstance(state_dist, (np.ndarray, list)):
            p = np.random.rand(self.num_states)
            p /= p.sum()
        else:
            assert len(state_dist) == self.num_states
            p = np.array(state_dist)

        if density == -1.0:
            self.density = self.num_nodes

        x = np.zeros([self.num_nodes, self.num_states])

        if isinstance(density, (float, int)):
            num_particules = np.random.poisson(density, size=self.num_nodes)
        elif isinstance(density, (list, np.ndarray)):
            assert len(density) == self.num_nodes
            num_particules = np.array(density)

        for i, n in enumerate(num_particules):
            x[i] = np.random.multinomial(n, p)

        return x

    def loglikelihood(self, x):
        return 1

    def predict(self, x):
        if len(x.shape) == 3:
            x = x[:, :, -1].squeeze()
        y = np.zeros(x.shape)

        p_reaction = self.reaction(x)
        for v in range(self.num_nodes):
            y[v] = x[v] @ p_reaction[v]

        p_diffusion = self.diffusion(y)
        for (v1, v2) in self.network.edges():
            n = p_diffusion[v1, v2] * y[v2]
            for i, nn in enumerate(n):
                if nn > 0:
                    y[v1, i] += nn
                    y[v2, i] -= nn

        return y

    def sample(self, x):
        if len(x.shape) == 3:
            x = x[:, :, -1]
        y = x.copy().astype("int")
        p_reaction = self.reaction(x)
        for v in range(self.num_nodes):
            for i, j in product(range(self.num_states), range(self.num_states)):
                n = int(np.random.binomial(int(x[v, j]), p_reaction[v, j, i]))
                y[v, i] += n
                y[v, j] -= n

        p_diffusion = self.diffusion(y)
        for (v1, v2) in self.network.edges():
            n = np.random.binomial(y[v2], p_diffusion[v1, v2]).astype("int")
            y[v1] += n
            y[v2] -= n

        return y.astype("int")

    def is_dead(self, x):
        if np.all(x[:, 1] == 0):
            return True
        else:
            return False


class WeightedMetaPop(MetaPop, WeightedDynamics):
    def __init__(self, config, num_states):
        MetaPop.__init__(self, config, num_states)
        WeightedDynamics.__init__(self, config, num_states)


class MultiplexMetaPop(MetaPop, MultiplexDynamics):
    def __init__(self, config, num_states):
        MetaPop.__init__(self, config, num_states)
        MultiplexDynamics.__init__(self, config, num_states)

    def predict(self, x):
        if len(x.shape) == 3:
            x = x[:, :, -1].squeeze()
        y = np.zeros(x.shape)

        p_reaction = self.reaction(x)
        for v in range(self.num_nodes):
            y[v] = x[v] @ p_reaction[v]

        p_diffusion = self.diffusion(y)
        for (v1, v2) in self.network["all"].edges():
            n = p_diffusion[v1, v2] * y[v2]
            for i, nn in enumerate(n):
                if nn > 0:
                    y[v1, i] += nn
                    y[v2, i] -= nn

        return y

    def sample(self, x):
        if len(x.shape) == 3:
            x = x[:, :, -1]
        y = x.copy().astype("int")
        p_reaction = self.reaction(x)
        for v in range(self.num_nodes):
            for i, j in product(range(self.num_states), range(self.num_states)):
                n = int(np.random.binomial(int(x[v, j]), p_reaction[v, j, i]))
                y[v, i] += n
                y[v, j] -= n

        p_diffusion = self.diffusion(y)
        for (v1, v2) in self.network["all"].edges():
            n = np.random.binomial(y[v2], p_diffusion[v1, v2]).astype("int")
            y[v1] += n
            y[v2] -= n

        return y.astype("int")


class WeightedMultiplexMetaPop(MetaPop, WeightedMultiplexDynamics):
    def __init__(self, config, num_states):
        MetaPop.__init__(self, config, num_states)
        WeightedMultiplexDynamics.__init__(self, config, num_states)

    def predict(self, x):
        if len(x.shape) == 3:
            x = x[:, :, -1].squeeze()
        y = np.zeros(x.shape)

        p_reaction = self.reaction(x)
        for v in range(self.num_nodes):
            y[v] = x[v] @ p_reaction[v]

        p_diffusion = self.diffusion(y)
        for (v1, v2) in self.network["all"].edges():
            n = p_diffusion[v1, v2] * y[v2]
            for i, nn in enumerate(n):
                if nn > 0:
                    y[v1, i] += nn
                    y[v2, i] -= nn

        return y

    def sample(self, x):
        if len(x.shape) == 3:
            x = x[:, :, -1]
        y = x.copy().astype("int")
        p_reaction = self.reaction(x)
        for v in range(self.num_nodes):
            for i, j in product(range(self.num_states), range(self.num_states)):
                n = int(np.random.binomial(int(x[v, j]), p_reaction[v, j, i]))
                y[v, i] += n
                y[v, j] -= n

        p_diffusion = self.diffusion(y)
        for (v1, v2) in self.network["all"].edges():
            n = np.random.binomial(y[v2], p_diffusion[v1, v2]).astype("int")
            y[v1] += n
            y[v2] -= n

        return y.astype("int")
