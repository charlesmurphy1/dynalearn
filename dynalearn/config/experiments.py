import dynalearn as dl
import numpy as np
import time
import os

from dynalearn.config import *
from ._utils import TrainingConfig, CallbackConfig

network_config = {
    "er": NetworkConfig.erdosrenyi(),
    "uni_er": NetworkConfig.erdosrenyi(weights=NetworkWeightConfig.uniform()),
    "norm_er": NetworkConfig.erdosrenyi(weights=NetworkWeightConfig.normal()),
    "loguni_er": NetworkConfig.erdosrenyi(weights=NetworkWeightConfig.loguniform()),
    "lognorm_er": NetworkConfig.erdosrenyi(weights=NetworkWeightConfig.lognormal()),
    "ba": NetworkConfig.barabasialbert(p=-1),
    "uni_ba": NetworkConfig.barabasialbert(weights=NetworkWeightConfig.uniform()),
    "norm_ba": NetworkConfig.barabasialbert(weights=NetworkWeightConfig.normal()),
    "loguni_ba": NetworkConfig.barabasialbert(weights=NetworkWeightConfig.loguniform()),
    "lognorm_ba": NetworkConfig.barabasialbert(weights=NetworkWeightConfig.lognormal()),
    "uni_multi_ba": NetworkConfig.barabasialbert(
        weights=NetworkWeightConfig.uniform(), num_layers=5
    ),
    "norm_multi_ba": NetworkConfig.barabasialbert(
        weights=NetworkWeightConfig.normal(), num_layers=5
    ),
    "loguni_multi_ba": NetworkConfig.barabasialbert(
        weights=NetworkWeightConfig.loguniform(), num_layers=5
    ),
    "lognorm_multi_ba": NetworkConfig.barabasialbert(
        weights=NetworkWeightConfig.lognormal(), num_layers=5
    ),
    "treeba": NetworkConfig.barabasialbert(m=1),
}
dynamics_config = {
    "sis": DynamicsConfig.sis(),
    "plancksis": DynamicsConfig.plancksis(),
    "sissis": DynamicsConfig.sissis(),
    "dsir": DynamicsConfig.dsir(),
}
model_config = {
    "sis": TrainableConfig.sis(),
    "plancksis": TrainableConfig.plancksis(),
    "sissis": TrainableConfig.sissis(),
    "dsir": TrainableConfig.dsir(),
}
metrics_config = {
    "sis": MetricsConfig.sis(),
    "plancksis": MetricsConfig.plancksis(),
    "sissis": MetricsConfig.sissis(),
    "dsir": MetricsConfig.dsir(),
}
trainingmetrics = {
    "sis": ["jensenshannon", "model_entropy"],
    "plancksis": ["jensenshannon", "model_entropy"],
    "sissis": ["jensenshannon", "model_entropy"],
    "dsir": ["jensenshannon", "model_entropy"],
}


class ExperimentConfig(Config):
    @classmethod
    def default(
        cls,
        name,
        dynamics,
        network,
        path_to_data="./",
        path_to_best="./",
        path_to_summary="./",
        seed=None,
    ):
        cls = cls()
        if dynamics not in dynamics_config:
            raise ValueError(
                f"{dynamics} is invalid, valid entries are {list(dynamics_config.keys())}"
            )
        if network not in network_config:
            raise ValueError(
                f"{network} is invalid, valid entries are {list(network_config.keys())}"
            )
        cls.name = name

        cls.path_to_data = os.path.join(path_to_data, cls.name)
        if not os.path.exists(cls.path_to_data):
            os.makedirs(cls.path_to_data)

        cls.path_to_best = os.path.join(path_to_best, cls.name + ".pt")
        if not os.path.exists(path_to_best):
            os.makedirs(path_to_best)

        cls.path_to_summary = path_to_summary
        if not os.path.exists(path_to_summary):
            os.makedirs(path_to_summary)
        cls.dynamics = dynamics_config[dynamics]
        cls.networks = network_config[network]
        cls.model = model_config[dynamics]
        if cls.networks.is_weighted:
            cls.dynamics.is_weighted = True
            cls.model.is_weighted = True
        else:
            cls.dynamics.is_weighted = False
            cls.model.is_weighted = False
        if cls.networks.is_multiplex:
            cls.dynamics.is_multiplex = True
            cls.model.is_multiplex = True
            cls.model.network_layers = cls.networks.layers
            cls.model.network_layers.append("all")
        else:
            cls.dynamics.is_multiplex = False
            cls.model.is_multiplex = False

        cls.metrics = metrics_config[dynamics]
        cls.train_metrics = trainingmetrics[dynamics]
        cls.callbacks = CallbackConfig.default(cls.path_to_best)

        if seed is None:
            cls.seed = int(time.time())
        else:
            cls.seed = seed

        return cls

    @classmethod
    def stocont(
        cls,
        name,
        dynamics,
        network,
        path_to_data="./",
        path_to_best="./",
        path_to_summary="./",
        seed=None,
    ):
        cls = cls.default(
            name,
            dynamics,
            network,
            path_to_data=path_to_data,
            path_to_best=path_to_best,
            path_to_summary=path_to_summary,
            seed=None,
        )
        cls.dataset = DiscreteDatasetConfig.state()
        cls.train_details = TrainingConfig.discrete()

        return cls

    @classmethod
    def metapop(
        cls,
        name,
        dynamics,
        network,
        path_to_data="./",
        path_to_best="./",
        path_to_summary="./",
        seed=None,
    ):
        cls = cls.default(
            name,
            dynamics,
            network,
            path_to_data=path_to_data,
            path_to_best=path_to_best,
            path_to_summary=path_to_summary,
            seed=None,
        )
        cls.dataset = ContinuousDatasetConfig.state(
            compounded=False, reduce=False, total=True
        )
        cls.train_details = TrainingConfig.continuous()

        return cls

    @classmethod
    def test(
        cls, path_to_data="./", path_to_best="./", path_to_summary="./",
    ):
        cls = cls()
        cls.name = "test"
        cls.path_to_data = os.path.join(path_to_data, cls.name)
        if not os.path.exists(cls.path_to_data):
            os.makedirs(cls.path_to_data)
        cls.path_to_best = os.path.join(path_to_best, cls.name + ".pt")
        if not os.path.exists(path_to_best):
            os.makedirs(path_to_best)
        cls.path_to_summary = path_to_summary
        if not os.path.exists(path_to_summary):
            os.makedirs(path_to_summary)
        cls.dataset = DiscreteDatasetConfig.state()
        cls.networks = NetworkConfig.erdosrenyi(1000, 4.0 / 999.0)
        cls.dynamics = DynamicsConfig.sis()
        cls.model = TrainableConfig.sis()
        cls.train_details = TrainingConfig.test()
        cls.metrics = MetricsConfig.test()
        cls.train_metrics = []
        cls.callbacks = CallbackConfig.default(cls.path_to_best)

        cls.seed = 0

        return cls
