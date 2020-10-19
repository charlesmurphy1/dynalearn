import networkx as nx
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
        if not isinstance(init_param, (np.ndarray, list)):
            p = np.random.rand(self.num_states)
            p /= p.sum()
        else:
            assert len(init_param) == self.num_states
            p = np.array(init_param)

        x = np.zeros([self.num_nodes, self.num_states])
        self.population = self.init_population(density=density)

        for i, n in enumerate(self.population):
            x[i] = np.random.multinomial(n, p) / n

        return x

    def init_population(self, density=None):
        if density is None:
            density = self.density
        if density == -1.0:
            density = self.num_nodes
        if isinstance(self.network, dict):
            g = self.network["all"]
        else:
            g = self.network
        assert isinstance(g, nx.Graph)
        if "population" in g.nodes[0]:
            population = get_node_attr(g)["population"]
        else:
            if isinstance(density, (float, int)):
                population = np.random.poisson(density, size=self.num_nodes)
            elif isinstance(density, (list, np.ndarray)):
                assert len(density) == self.num_nodes
                population = np.array(density)
        if isinstance(self.network, dict):
            self._network["all"] = set_node_attr(g, {"population": population})
        else:
            self._network = set_node_attr(g, {"population": population})
        return population

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

    def update_node_attr(self):
        self.population = self.init_population()


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
