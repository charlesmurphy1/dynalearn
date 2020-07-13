import numpy as np

from dynalearn.dynamics.dynamics import Dynamics
from abc import abstractmethod
from itertools import product


class MetaPopulationDynamics(Dynamics):
    def __init__(self, config, num_states):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        self.initial_state_dist = config.initial_state_dist
        self.initial_density = config.initial_density
        Dynamics.__init__(self, config, num_states)

    @abstractmethod
    def reaction(self, x):
        raise NotImplemented()

    @abstractmethod
    def diffusion(self, x):
        raise NotImplemented()

    def initial_state(self, initial_state_dist=None, initial_density=None):
        if initial_state_dist is None:
            initial_state_dist = self.initial_state_dist
        if initial_density is None:
            initial_density = self.initial_density

        if initial_state_dist == -1.0:
            p = np.random.rand(self.num_states)
            p /= p.sum()
        else:
            p = np.array(initial_state_dist)

        x = np.zeros([self.num_nodes, self.num_states])
        num_particules = np.random.poisson(self.initial_density, size=self.num_nodes)
        for i, n in enumerate(num_particules):
            x[i] = np.random.multinomial(n, p)

        return x

    def predict(self, x):
        if len(x.shape) == 3:
            x = x[-1]
        y = np.zeros(x.shape)

        p_reaction = self.reaction(x)
        for v in range(self.num_nodes):
            y[v] = x[v] @ p_reaction[v]

        p_diffusion = self.diffusion(y)
        for (v1, v2) in self.network.edges():
            n = p_diffusion[v1, v2] * y[v2]
            y[v1] += n
            y[v2] -= n

        return y

    def sample(self, x):
        if len(x.shape) == 3:
            x = x[-1]
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
