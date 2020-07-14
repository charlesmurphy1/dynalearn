import dynalearn as dl
import numpy as np
import time
import os

from dynalearn.config import *

network_config = {"er": NetworkConfig.er_default(), "ba": NetworkConfig.ba_default()}
dynamics_config = {
    "sis": DynamicsConfig.sis_default(),
    "plancksis": DynamicsConfig.plancksis_default(),
    "sissis": DynamicsConfig.sissis_default(),
    "hiddensissis": DynamicsConfig.hidden_sissis_default(),
}
model_config = {
    "sis": DynamicsConfig.sis_gnn_default(),
    "plancksis": DynamicsConfig.plancksis_gnn_default(),
    "sissis": DynamicsConfig.sissis_gnn_default(),
    "hiddensissis": DynamicsConfig.hidden_sissis_gnn_default(),
}
metrics_config = {
    "sis": MetricsConfig.sis(),
    "plancksis": MetricsConfig.plancksis(),
    "sissis": MetricsConfig.sissis(),
    "hiddensissis": MetricsConfig.hidden_sissis(),
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
        cls.threshold_window_size = 3

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
        cls.threshold_window_size = 3

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
    def discrete_experiment(
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
        cls.model = model_config[dynamics]
        cls.networks = network_config[network]

        if dynamics == "hiddensissis":
            cls.dataset = DatasetConfig.state_weighted_hidden_sissis()
            # if cls.model.window_size > 5:
            #    cls.dataset = DatasetConfig.degree_weighted_hidden_sissis()
        else:
            cls.dataset = DatasetConfig.state_weighted_discrete_default()
        cls.train_details = TrainingConfig.default()
        cls.metrics = metrics_config[dynamics]
        cls.train_metrics = ["jensenshannon", "model_entropy"]
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
    def test(cls, path_to_data="./", path_to_best="./", path_to_summary="./"):
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
        cls.dataset = DatasetConfig.state_weighted_discrete_default()
        cls.networks = NetworkConfig.erdosrenyi(1000, 4.0 / 999.0)
        cls.dynamics = DynamicsConfig.sis_default()
        cls.model = DynamicsConfig.sis_gnn_default()

        cls.train_details = TrainingConfig.test()
        cls.metrics = MetricsConfig.test()
        cls.train_metrics = []
        cls.callbacks = CallbackConfig.default(cls.path_to_best)

        cls.seed = 0

        return cls
