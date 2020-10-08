import numpy as np

from abc import abstractmethod
from itertools import product
from dynalearn.dynamics.dynamics import (
    Dynamics,
    WeightedDynamics,
    MultiplexDynamics,
    WeightedMultiplexDynamics,
)
from dynalearn.utilities import set_node_attr, get_node_attr


class DeterministicEpidemics(Dynamics):
    def __init__(self, config, num_states):
        if "init_param" in config.__dict__:
            self.init_param = config.init_param
        else:
            self.init_param = -1
        if "density" in config.__dict__:
            self.density = config.density
        else:
            self.density = -1

        self.population = None
        Dynamics.__init__(self, config, num_states)

    @abstractmethod
    def update(self, x):
        raise NotImplemented()

    def initial_state(self, init_param=None, density=None):
        if init_param is None:
            init_param = self.init_param
        if density is None:
            density = self.density
        if density == -1.0:
            density = self.num_nodes
        if not isinstance(init_param, (np.ndarray, list)):
            p = np.random.rand(self.num_states)
            p /= p.sum()
        else:
            assert len(init_param) == self.num_states
            p = np.array(init_param)

        x = np.zeros([self.num_nodes, self.num_states])
        if isinstance(self.network, dict):
            g = self.network["all"]
        else:
            g = self.network
        if "population" not in g.nodes[0]:
            if isinstance(density, (float, int)):
                self.population = np.random.poisson(density, size=self.num_nodes)
            elif isinstance(density, (list, np.ndarray)):
                assert len(density) == self.num_nodes
                self.population = np.array(density)
            g = set_node_attr(g, {"population": self.population})

        if isinstance(self.network, dict):
            self.network["all"] = g
        else:
            self.network = g

        for i, n in enumerate(self.population):
            x[i] = np.random.multinomial(n, p) / n

        return x

    def loglikelihood(self, x):
        return 1

    def predict(self, x):
        if len(x.shape) == 3:
            x = x[:, :, -1].squeeze()
        dx = self.update(x)
        y = x + dx
        y[y < 0] = 0
        y[y > 1] = 1
        y /= y.sum(-1, keepdims=True)

        return y

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
