import numpy as np

from .config import Config


class DynamicsConfig(Config):
    @classmethod
    def sis(cls):
        cls = cls()
        cls.name = "SIS"
        cls.infection = 0.04
        cls.recovery = 0.08
        cls.init_param = None
        return cls

    @classmethod
    def sir(cls):
        cls = cls()
        cls.name = "SIR"
        cls.infection = 0.04
        cls.recovery = 0.08
        cls.init_param = None
        return cls

    @classmethod
    def plancksis(cls):
        cls = cls()
        cls.name = "PlanckSIS"
        cls.temperature = 6.0
        cls.recovery = 0.08
        cls.init_param = None

        return cls

    @classmethod
    def sissis(cls):
        cls = cls()
        cls.name = "SISSIS"
        cls.infection1 = 0.02
        cls.infection2 = 0.01
        cls.recovery1 = 0.12
        cls.recovery2 = 0.13
        cls.coupling = 10.0
        cls.init_param = None

        return cls

    @classmethod
    def dsir(cls):
        cls = cls()
        cls.name = "DSIR"
        cls.infection_prob = 2.5 / 2.3
        cls.recovery_prob = 1.0 / 7.5
        cls.infection_type = 2
        cls.density = 10000
        epsilon = 1e-5
        cls.init_param = np.array([1 - epsilon, epsilon, 0])
        return cls

    @classmethod
    def dsir_covid(cls):
        cls = cls()
        cls.name = "DSIR"
        cls.infection_prob = 2.5 / 2.3
        cls.recovery_prob = 1.0 / (7.5)
        cls.infection_type = 2
        cls.density = array(
            [
                331549.0,
                388167.0,
                1858683.0,
                716820.0,
                157640.0,
                673559.0,
                1149460.0,
                5664579.0,
                356958.0,
                394151.0,
                1240155.0,
                579962.0,
                495761.0,
                782979.0,
                1119596.0,
                196329.0,
                771044.0,
                914678.0,
                257762.0,
                723576.0,
                521870.0,
                220461.0,
                633564.0,
                460001.0,
                434930.0,
                316798.0,
                329587.0,
                6663394.0,
                1661785.0,
                1493898.0,
                654214.0,
                307651.0,
                1022800.0,
                160980.0,
                1120406.0,
                942665.0,
                330119.0,
                1032983.0,
                581078.0,
                153129.0,
                1942389.0,
                88636.0,
                804664.0,
                134137.0,
                694844.0,
                2565124.0,
                519546.0,
                1152651.0,
                172539.0,
                964693.0,
                84777.0,
                86487.0,
            ]
        )
        cls.init_param = None
        return cls
