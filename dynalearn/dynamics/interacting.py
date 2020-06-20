import numpy as np
import torch

from dynalearn.dynamics.epidemics import MultiEpidemics
from dynalearn.config import Config


class SISSIS(MultiEpidemics):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        num_diseases = 2
        num_states = 4

        self.infection1 = config.infection1
        self.infection2 = config.infection2
        self.recovery1 = config.recovery1
        self.recovery2 = config.recovery2
        self.coupling = config.coupling

        super(SISSIS, self).__init__(config, num_diseases, num_states)

    def predict(self, x):
        l = self.neighbors_state(x)
        p0, p1 = self.infection(x, l)
        q0, q1 = self.recovery(x, l)

        ltp = np.zeros((x.shape[0], self.num_states))

        # SS nodes
        ltp[x == 0, 0] = (1 - p0[x == 0]) * (1 - p1[x == 0])
        ltp[x == 0, 1] = p0[x == 0] * (1 - p1[x == 0])
        ltp[x == 0, 2] = (1 - p0[x == 0]) * p1[x == 0]
        ltp[x == 0, 3] = p0[x == 0] * p1[x == 0]

        # IS nodes
        ltp[x == 1, 0] = q0[x == 1] * (1 - p1[x == 1])
        ltp[x == 1, 1] = (1 - q0[x == 1]) * (1 - p1[x == 1])
        ltp[x == 1, 2] = q0[x == 1] * p1[x == 1]
        ltp[x == 1, 3] = (1 - q0[x == 1]) * p1[x == 1]

        # SI nodes
        ltp[x == 2, 0] = (1 - p0[x == 2]) * q1[x == 2]
        ltp[x == 2, 1] = p0[x == 2] * q1[x == 2]
        ltp[x == 2, 2] = (1 - p0[x == 2]) * (1 - q1[x == 2])
        ltp[x == 2, 3] = p0[x == 2] * (1 - q1[x == 2])

        # II nodes
        ltp[x == 3, 0] = q0[x == 3] * q1[x == 3]
        ltp[x == 3, 1] = (1 - q0[x == 3]) * q1[x == 3]
        ltp[x == 3, 2] = q0[x == 3] * (1 - q1[x == 3])
        ltp[x == 3, 3] = (1 - q0[x == 3]) * (1 - q1[x == 3])

        return ltp

    def infection(self, x, l):

        inf0 = np.zeros(x.shape)
        inf1 = np.zeros(x.shape)

        # Node SS
        inf0[x == 0] = (
            1
            - (1 - self.infection1) ** l[1, x == 0]
            * (1 - self.coupling * self.infection1) ** l[3, x == 0]
        )
        inf1[x == 0] = (
            1
            - (1 - self.infection2) ** l[2, x == 0]
            * (1 - self.coupling * self.infection2) ** l[3, x == 0]
        )

        # Node IS
        inf1[x == 1] = 1 - (1 - self.coupling * self.infection2) ** (
            l[2, x == 1] + l[3, x == 1]
        )

        # Node SI
        inf0[x == 2] = 1 - (1 - self.coupling * self.infection1) ** (
            l[1, x == 2] + l[3, x == 2]
        )
        return inf0, inf1

    def recovery(self, x, l):
        rec0 = np.ones(x.shape) * self.recovery1
        rec1 = np.ones(x.shape) * self.recovery2

        return rec0, rec1


class HiddenSISSIS(SISSIS):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        SISSIS.__init__(self, config, **kwargs)
        self.state_map = {0: 0, 1: 1, 2: 0, 3: 1}

    def predict(self, x):
        p = SISSIS.predict(self, x)

        ltp = np.zeros((x.shape[0], 2))
        ltp[:, 0] = p[:, 0] + p[:, 2]
        ltp[:, 1] = p[:, 1] + p[:, 3]

        return ltp

    def sample(self, x):
        p = SISSIS.predict(self, x)
        dist = torch.distributions.Categorical(torch.tensor(p))
        x = np.array(dist.sample())
        return np.vectorize(self.state_map.get)(x)
