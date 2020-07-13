import numpy as np

from dynalearn.dynamics.metapopulation import MetaPopulationDynamics
from dynalearn.config import Config


class MetaSIS(MetaPopulationDynamics):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        MetaPopulationDynamics.__init__(self, config, 2)
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
        k = dict(self.network.degree())
        for (i, j) in self.network.edges():
            p[i, j] = self.diffusion_prob / k[j]
        return p

    def is_dead(self, x):
        if np.all(x[:, 1] == 0):
            return True
        else:
            return False
