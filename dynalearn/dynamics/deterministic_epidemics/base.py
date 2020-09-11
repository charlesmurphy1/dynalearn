import numpy as np

from abc import abstractmethod
from itertools import product
from dynalearn.dynamics.dynamics import (
    Dynamics,
    WeightedDynamics,
    MultiplexDynamics,
    WeightedMultiplexDynamics,
)
from dynalearn.utilities import set_node_attr


class DeterministicEpidemics(Dynamics):
    def __init__(self, config, num_states):
        if "state_dist" in config.__dict__:
            self.state_dist = config.state_dist
        else:
            self.state_dist = -1
        if "density" in config.__dict__:
            self.density = config.density
        else:
            self.density = -1

        self.population = None
        Dynamics.__init__(self, config, num_states)

    @abstractmethod
    def update(self, x):
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

        if "population" not in self.network.nodes[0]:
            if self.population is None:
                if isinstance(density, (float, int)):
                    self.population = np.random.poisson(density, size=self.num_nodes)
                elif isinstance(density, (list, np.ndarray)):
                    assert len(density) == self.num_nodes
                    self.population = np.array(density)
            self.network = set_node_attr(self.network, {"population": self.population})
        for i, n in enumerate(self.population):
            x[i] = np.random.multinomial(n, p) / n

        return x

    def loglikelihood(self, x):
        return 1

    def predict(self, x):
        if len(x.shape) == 3:
            x = x[:, :, -1].squeeze()
        return x + self.update(x)

    def sample(self, x):
        return self.predict(x)

    def is_dead(self, x):
        if np.all(x[:, 1] == 0):
            return True
        else:
            return False


class WeightedDeterministicEpidemics(DeterministicEpidemics, WeightedDynamics):
    def __init__(self, config, num_states):
        DeterministicEpidemics.__init__(self, config, num_states)
        WeightedDynamics.__init__(self, config, num_states)


class MultiplexDeterministicEpidemics(DeterministicEpidemics, MultiplexDynamics):
    def __init__(self, config, num_states):
        DeterministicEpidemics.__init__(self, config, num_states)
        MultiplexDynamics.__init__(self, config, num_states)


class WeightedMultiplexDeterministicEpidemics(
    DeterministicEpidemics, WeightedMultiplexDynamics
):
    def __init__(self, config, num_states):
        DeterministicEpidemics.__init__(self, config, num_states)
        WeightedMultiplexDynamics.__init__(self, config, num_states)
