import dynalearn as dl
import numpy as np
import time
import os

from dynalearn.config import *

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
        weights=NetworkWeightConfig.uniform(), num_layers=2
    ),
    "norm_multi_ba": NetworkConfig.barabasialbert(
        weights=NetworkWeightConfig.normal(), num_layers=2
    ),
    "loguni_multi_ba": NetworkConfig.barabasialbert(
        weights=NetworkWeightConfig.loguniform(), num_layers=2
    ),
    "lognorm_multi_ba": NetworkConfig.barabasialbert(
        weights=NetworkWeightConfig.lognormal(), num_layers=2
    ),
    "treeba": NetworkConfig.barabasialbert(m=1),
}
dynamics_config = {
    "sis": DynamicsConfig.sis(),
    "plancksis": DynamicsConfig.plancksis(),
    "sissis": DynamicsConfig.sissis(),
    "hiddensissis": DynamicsConfig.hidden_sissis(),
    "partiallyhiddensissis": DynamicsConfig.partially_hidden_sissis(),
    "rdsis": DynamicsConfig.rdsis(),
    "rdsir": DynamicsConfig.rdsir(),
    "dsis": DynamicsConfig.dsis(),
    "dsir": DynamicsConfig.dsir(),
}
model_config = {
    "sis": TrainableConfig.sis(),
    "plancksis": TrainableConfig.plancksis(),
    "sissis": TrainableConfig.sissis(),
    "hiddensissis": TrainableConfig.hidden_sissis(),
    "partiallyhiddensissis": TrainableConfig.partially_hidden_sissis(),
    "rdsis": TrainableConfig.rdsis(),
    "rdsir": TrainableConfig.rdsir(),
    "dsis": TrainableConfig.dsis(),
    "dsir": TrainableConfig.dsir(),
}
metrics_config = {
    "sis": MetricsConfig.sis(),
    "plancksis": MetricsConfig.plancksis(),
    "sissis": MetricsConfig.sissis(),
    "hiddensissis": MetricsConfig.hidden_sissis(),
    "partiallyhiddensissis": MetricsConfig.partially_hidden_sissis(),
    "rdsis": MetricsConfig.rdsis(),
    "rdsir": MetricsConfig.rdsir(),
    "dsis": MetricsConfig.dsis(),
    "dsir": MetricsConfig.dsir(),
}
trainingmetrics = {
    "sis": ["acc", "jensenshannon", "model_entropy"],
    "plancksis": ["acc", "jensenshannon", "model_entropy"],
    "sissis": ["acc", "jensenshannon", "model_entropy"],
    "hiddensissis": ["acc", "jensenshannon", "model_entropy"],
    "partiallyhiddensissis": ["acc", "jensenshannon", "model_entropy"],
    "rdsis": ["acc"],
    "rdsir": ["acc"],
    "dsis": ["acc", "jensenshannon", "model_entropy"],
    "dsir": ["acc", "jensenshannon", "model_entropy"],
}


class TrainingConfig(Config):
    @classmethod
    def default(cls,):
        cls = cls()

        cls.val_fraction = 0.01
        cls.val_bias = 0.8
        cls.epochs = 30
        cls.batch_size = 32
        cls.num_nodes = 1000
        cls.num_networks = 1
        cls.num_samples = 10000
        cls.resampling = 2
        cls.max_window_size = 3
        cls.resample_when_dead = True

        return cls

    @classmethod
    def discrete(cls,):
        cls = cls()

        cls.val_fraction = 0.01
        cls.val_bias = 0.8
        cls.epochs = 30
        cls.batch_size = 32
        cls.num_networks = 1
        cls.num_samples = 10000
        cls.resampling = 2
        cls.max_window_size = 3
        cls.resample_when_dead = True

        return cls

    @classmethod
    def continuous(cls,):
        cls = cls()

        cls.val_fraction = 0.1
        cls.val_bias = 0.5
        cls.epochs = 30
        cls.batch_size = 32
        cls.num_networks = 1
        cls.num_samples = 10000
        cls.resampling = 2
        cls.max_window_size = 3
        cls.resample_when_dead = False

        return cls

    @classmethod
    def test(cls,):
        cls = cls()

        cls.val_fraction = 0.01
        cls.val_bias = 0.8
        cls.epochs = 5
        cls.batch_size = 10
        cls.num_networks = 1
        cls.num_samples = 10
        cls.resampling = 2
        cls.max_window_size = 3
        cls.resample_when_dead = True

        return cls


class CallbackConfig(Config):
    @classmethod
    def default(cls, path_to_best="./"):
        cls = cls()
        cls.names = ["ModelCheckpoint", "StepLR"]
        cls.step_size = 10
        cls.gamma = 0.5
        cls.path_to_best = path_to_best
        return cls

    @classmethod
    def empty(cls):
        cls = cls()
        cls.names = []
        return cls


class ExperimentConfig(Config):
    @classmethod
    def discrete(
        cls,
        name,
        dynamics,
        network,
        path_to_data="./",
        path_to_best="./",
        path_to_summary="./",
        mode="fast",
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

        if dynamics == "hiddensissis":
            cls.dataset = DiscreteDatasetConfig.hidden_sissis()
        elif dynamics == "partiallyhiddensissis":
            cls.dataset = DiscreteDatasetConfig.partially_hidden_sissis()
        else:
            cls.dataset = DiscreteDatasetConfig.state()
        cls.train_details = TrainingConfig.discrete()
        cls.metrics = metrics_config[dynamics]
        cls.train_metrics = ["jensenshannon", "model_entropy"]
        cls.callbacks = CallbackConfig.default(cls.path_to_best)

        if seed is None:
            cls.seed = int(time.time())
        else:
            cls.seed = seed

        return cls

    @classmethod
    def continuous(
        cls,
        name,
        dynamics,
        network,
        path_to_data="./",
        path_to_best="./",
        path_to_summary="./",
        mode="fast",
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
        else:
            cls.dynamics.is_multiplex = False
            cls.model.is_multiplex = False

        cls.dataset = ContinuousDatasetConfig.state(
            compounded=False, reduce=False, total=True
        )
        cls.train_details = TrainingConfig.continuous()
        cls.metrics = metrics_config[dynamics]
        cls.train_metrics = trainingmetrics[dynamics]
        cls.callbacks = CallbackConfig.default(cls.path_to_best)

        if seed is None:
            cls.seed = int(time.time())
        else:
            cls.seed = seed

        return cls

    @classmethod
    def spain_covid19(
        cls,
        path_to_data="./",
        path_to_best="./",
        path_to_summary="./",
        weighted=False,
        multiplex=False,
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
        cls.name = "spain_covid19"

        cls.path_to_data = os.path.join(path_to_data, cls.name)
        if not os.path.exists(cls.path_to_data):
            os.makedirs(cls.path_to_data)

        cls.path_to_best = os.path.join(path_to_best, cls.name + ".pt")
        if not os.path.exists(path_to_best):
            os.makedirs(path_to_best)

        cls.path_to_summary = path_to_summary
        if not os.path.exists(path_to_summary):
            os.makedirs(path_to_summary)
        cls.dynamics = DynamicsConfig.metasir_covid19()
        cls.networks = NetworkConfig.spain_mobility(path_to_data)
        cls.model = DynamicsConfig.metasir_gnn_covid19()
        cls.dynamics.is_weighted = weighted
        cls.model.is_weighted = weighted

        cls.dynamics.is_multiplex = multiplex
        cls.model.is_multiplex = multiplex

        cls.dataset = DatasetConfig.strength_weighted_continuous_default()
        cls.train_details = TrainingConfig.continuous()
        cls.metrics = MetricsConfig.covid19()
        cls.train_metrics = ["acc"]
        cls.callbacks = CallbackConfig.default(cls.path_to_best)

        if seed is None:
            cls.seed = int(time.time())
        else:
            cls.seed = seed

        return cls

    @classmethod
    def rtn_forcast(
        cls,
        name,
        dynamics,
        path_to_edgelist,
        path_to_model,
        path_to_data="./",
        path_to_summary="./",
        seed=None,
    ):
        cls = cls()
        if name is None:
            cls.name = f"{dynamics}-rtn"
        else:
            cls.name = name

        cls.path_to_data = os.path.join(path_to_data, cls.name)
        if not os.path.exists(cls.path_to_data):
            os.makedirs(cls.path_to_data)

        cls.path_to_best = path_to_model
        cls.fname_best = f"{dynamics}-ba-ns10000.pt"

        cls.path_to_summary = path_to_summary
        if not os.path.exists(cls.path_to_summary):
            os.makedirs(cls.path_to_summary)

        cls.dataset = DatasetConfig.plain_default()
        cls.networks = NetworkConfig.realtemporalnetwork(path_to_edgelist, window=12)
        cls.dynamics = dynamics_config[dynamics]
        cls.model = model_config[dynamics]

        cls.train_details = TrainingConfig.default()
        cls.metrics = MetricsConfig.rtn_forecast()
        cls.train_metrics = []
        cls.callbacks = CallbackConfig.empty()

        if seed is None:
            cls.seed = int(time.time())
        else:
            cls.seed = seed

        return cls

    @classmethod
    def test(
        cls,
        config="discrete",
        path_to_data="./",
        path_to_best="./",
        path_to_summary="./",
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
        if config == "discrete":
            cls.dataset = DiscreteDatasetConfig.state()
            cls.networks = NetworkConfig.erdosrenyi(1000, 4.0 / 999.0)
            cls.dynamics = DynamicsConfig.sis_default()
            cls.model = DynamicsConfig.sis_gnn_default()
        elif config == "continuous":
            cls.dataset = ContinuousDatasetConfig.state()
            cls.networks = NetworkConfig.erdosrenyi(1000, 4.0 / 999.0)
            cls.dynamics = DynamicsConfig.metasis_default()
            cls.model = DynamicsConfig.metasis_gnn_default()
        cls.train_details = TrainingConfig.test()
        cls.metrics = MetricsConfig.test()
        cls.train_metrics = []
        cls.callbacks = CallbackConfig.default(cls.path_to_best)

        cls.seed = 0

        return cls
