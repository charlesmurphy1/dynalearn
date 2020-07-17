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
        if len(x.shape) > 1:
            x = x[-1].squeeze()
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


class AsymmetricSISSIS(SISSIS):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        SISSIS.__init__(self, config, **kwargs)
        boost = config.boost
        if boost == "source":
            self.infection = self._source_infection_
        elif boost == "target":
            self.infection = self._target_infection_
        else:
            raise ValueError(
                f"{boost} is invalid, valid entries are ['source', 'target']"
            )

    def _source_infection_(self, x, l):
        inf0 = np.zeros(x.shape)
        inf1 = np.zeros(x.shape)

        # Node SS
        inf0[x == 0] = 1 - (1 - self.infection1) ** (l[1, x == 0] + l[3, x == 0])
        inf1[x == 0] = 1 - (1 - self.infection2) ** (l[2, x == 0] + l[3, x == 0])

        # Node IS
        inf1[x == 1] = 1 - (1 - self.coupling * self.infection2) ** (
            l[2, x == 1] + l[3, x == 1]
        )

        # Node SI
        inf0[x == 2] = 1 - (1 - self.coupling * self.infection1) ** (
            l[1, x == 2] + l[3, x == 2]
        )
        return inf0, inf1

    def _target_infection_(self, x, l):
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
        inf1[x == 1] = (
            1
            - (1 - self.infection2) ** l[2, x == 0]
            * (1 - self.coupling * self.infection2) ** l[3, x == 0]
        )

        # Node SI
        inf0[x == 2] = (
            1
            - (1 - self.infection1) ** l[1, x == 0]
            * (1 - self.coupling * self.infection1) ** l[3, x == 0]
        )
        return inf0, inf1


class HiddenSISSIS(SISSIS):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        SISSIS.__init__(self, config, **kwargs)
        self.state_map = {0: 0, 1: 1, 2: 0, 3: 1}
        self.hide = True

    def predict(self, x):
        if len(x.shape) > 1:
            x = x[-1].squeeze()
        p = SISSIS.predict(self, x)

        if self.hide:
            ltp = np.zeros((x.shape[0], 2))
            ltp[:, 0] = p[:, 0] + p[:, 2]
            ltp[:, 1] = p[:, 1] + p[:, 3]
            return ltp

        return p

    def sample(self, x):
        p = SISSIS.predict(self, x)
        dist = torch.distributions.Categorical(torch.tensor(p))
        x = np.array(dist.sample())
        return x


class PartiallyHiddenSISSIS(SISSIS):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        SISSIS.__init__(self, config, **kwargs)
        self.state_map = {0: 0, 1: 1, 2: 0, 3: 1}
        self.hide_prob = config.hide_prob
        self.hide = True

    def predict(self, x):
        if len(x.shape) > 1:
            x = x[-1].squeeze()
        p = SISSIS.predict(self, x)

        if self.hide:
            p[:, 2] = self.hide_prob * p[:, 0] + (1 - self.hide_prob) * p[:, 2]
            p[:, 3] = self.hide_prob * p[:, 1] + (1 - self.hide_prob) * p[:, 3]

        return p
