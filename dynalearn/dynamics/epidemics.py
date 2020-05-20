import networkx as nx
import numpy as np
import torch as pt

from dynalearn.dynamics.base import DynamicsModel
from dynalearn.nn.models import Propagator
from dynalearn.utilities import from_binary


class Epidemics(DynamicsModel):
    def __init__(self, config, num_states):
        DynamicsModel.__init__(self, config, num_states)
        self.initial_infected = config.initial_infected
        self.propagator = Propagator(num_states)

    def sample(self, x):
        p = self.predict(x)
        dist = pt.distributions.Categorical(pt.tensor(p))
        return np.array(dist.sample())

    def neighbors_state(self, x):
        if len(x.shape) > 1:
            raise ValueError(
                f"Invalid shape, expected shape of size 1 and got ({x.shape})"
            )

        l = self.propagator.forward(x, self.edge_index)
        l = l.cpu().numpy()
        return l


class MultiEpidemics(Epidemics):
    def __init__(self, params, num_diseases, num_states):
        self.num_diseases = num_diseases
        if num_states < 2:
            raise ValueError(
                f"num_states must be greater than or equal to {2**num_diseases}"
            )
        Epidemics.__init__(self, params, num_states)

    def initial_state(self, initial_infected=None):
        if initial_infected is None:
            initial_infected = self.initial_infected

        if initial_infected == -1.0:
            p = [np.random.rand() for i in range(self.num_diseases)]
        elif initial_infected >= 0 and initial_infected <= 1:
            p = np.ones(self.num_diseases) * (
                1 - (1 - initial_infected) ** (1.0 / self.num_diseases)
            )
        else:
            raise ValueError(
                "Value for 'initial_infected'"
                + f" must be between [0, 1] or equal to -1:"
                + f" Received {initial_infected}."
            )

        n_infected = [np.random.binomial(self.num_nodes, pp) for pp in p]
        nodeset = np.arange(self.num_nodes)
        index = [np.random.choice(nodeset, size=n, replace=False) for n in n_infected]
        bin_x = np.zeros((self.num_nodes, self.num_diseases))
        for i, ind in enumerate(index):
            bin_x[ind, i] = 1
        x = np.array([from_binary(b[::-1]) for b in bin_x])
        return x

    def is_dead(self, states):
        if np.all(states == 0):
            return True
        else:
            return False


class SingleEpidemics(MultiEpidemics):
    def __init__(self, config, num_states):
        MultiEpidemics.__init__(self, config, 1, num_states)
