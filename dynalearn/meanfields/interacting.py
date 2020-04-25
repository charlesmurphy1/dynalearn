import numpy as np
from .base import Meanfield
from dynalearn.dynamics.interacting import SISSIS
from dynalearn.meanfields import Meanfield
from dynalearn.utilities import Config


class SISSISMeanfield(Meanfield):
    def __init__(self, p_k, model):
        if model.__class__ is not SISSIS:
            raise ValueError(
                f"{model.__class__} is invalid, the model class must be 'SISSIS'."
            )
        self.model = model
        self.infection1 = model.infection1
        self.infection2 = model.infection2
        self.recovery1 = model.recovery1
        self.recovery2 = model.recovery2
        self.coupling = model.coupling
        Meanfield.__init__(self, p_k, 4, with_numba=False)

    def marginal_ltp(self, x, y, k, phi):
        inf1, inf2 = self.infection1, self.infection2
        rec1, rec2 = self.recovery1, self.recovery2
        c = self.coupling
        if x == 0:
            p = (phi[0] + phi[1] * (1 - inf1) + phi[2] + phi[3] * (1 - c * inf1)) ** k
            q = (phi[0] + phi[1] + phi[2] * (1 - inf2) + phi[3] * (1 - c * inf2)) ** k
            pq = (
                phi[0]
                + phi[1] * (1 - inf1)
                + phi[2] * (1 - inf2)
                + phi[3] * (1 - c * inf1) * (1 - c * inf2)
            ) ** k

            if y == 0:
                return pq
            elif y == 1:
                return q - pq
            elif y == 2:
                return p - pq
            elif y == 3:
                return 1 - p - q + pq
            raise ValueError(f"{y} for y is invalid. Entries must be [0, 1, 2, 3]")
        elif x == 1:
            p = (
                phi[0] + phi[1] + phi[2] * (1 - c * inf2) + phi[3] * (1 - c * inf2)
            ) ** k
            if y == 0:
                return (rec1) * p
            elif y == 1:
                return (1 - rec1) * p
            elif y == 2:
                return (rec1) * (1 - p)
            elif y == 3:
                return (1 - rec1) * (1 - p)
            raise ValueError(f"{y} for y is invalid. Entries must be [0, 1, 2, 3]")
        elif x == 2:
            p = (
                phi[0] + phi[1] * (1 - c * inf1) + phi[2] + phi[3] * (1 - c * inf1)
            ) ** k
            if y == 0:
                return p * (rec2)
            elif y == 1:
                return (1 - p) * (rec2)
            elif y == 2:
                return p * (1 - rec2)
            elif y == 3:
                return (1 - p) * (1 - rec2)
            raise ValueError(f"{y} for y is invalid. Entries must be [0, 1, 2, 3]")
        elif x == 3:
            if y == 0:
                return (rec1) * (rec2)
            elif y == 1:
                return (1 - rec1) * (rec2)
            elif y == 2:
                return (rec1) * (1 - rec2)
            elif y == 3:
                return (1 - rec1) * (1 - rec2)
            raise ValueError(f"{y} for y is invalid. Entries must be [0, 1, 2, 3]")

        raise ValueError(f"{x} for x is invalid. Entries must be [0, 1, 2, 3]")
