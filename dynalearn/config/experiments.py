import dynalearn as dl
import os

from dynalearn.config import *

network_config = {"er": NetworkConfig.er_default(), "ba": NetworkConfig.ba_default()}
dynamics_config = {
    "sis": DynamicsConfig.sis_default(),
    "plancksis": DynamicsConfig.plancksis_default(),
    "sissis": DynamicsConfig.sissis_default(),
}
model_config = {
    "sis": DynamicsConfig.sis_gnn_default(),
    "plancksis": DynamicsConfig.plancksis_gnn_default(),
    "sissis": DynamicsConfig.sissis_gnn_default(),
}
fast_metrics_config = {
    "sis": MetricsConfig.sis_fast(),
    "plancksis": MetricsConfig.plancksis_fast(),
    "sissis": MetricsConfig.sissis_fast(),
}
complete_metrics_config = {
    "sis": MetricsConfig.sis_complete(),
    "plancksis": MetricsConfig.plancksis_complete(),
    "sissis": MetricsConfig.sissis_complete(),
}
fast_summary_config = {
    "sis": SummariesConfig.sis_fast(),
    "plancksis": SummariesConfig.plancksis_fast(),
    "sissis": SummariesConfig.sissis_fast(),
}
complete_summary_config = {
    "sis": SummariesConfig.sis_complete(),
    "plancksis": SummariesConfig.plancksis_complete(),
    "sissis": SummariesConfig.sissis_complete(),
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
        cls.resampling_time = 2

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


class ExperimentConfig(Config):
    @classmethod
    def base(
        cls,
        name,
        dynamics,
        network,
        path_to_data="./",
        path_to_best="./",
        path_to_summary="./",
        mode="fast",
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
        cls.dataset = DatasetConfig.state_weighted_default()
        cls.dynamics = dynamics_config[dynamics]
        cls.model = model_config[dynamics]
        cls.networks = network_config[network]

        cls.train_details = TrainingConfig.default()
        if mode == "fast":
            cls.metrics = fast_metrics_config[dynamics]
            cls.summaries = fast_summary_config[dynamics]
        elif mode == "complete":
            cls.metrics = complete_metrics_config[dynamics]
            cls.summaries = complete_summary_config[dynamics]
        else:
            raise ValueError(
                f"{mode} is invalid, valid entries are ['fast', 'complete']."
            )
        cls.train_metrics = ["jensenshannon", "model_entropy"]
        cls.callbacks = CallbackConfig.default(cls.path_to_best)

        cls.seed = 0

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
        cls.dataset = DatasetConfig.state_weighted_default()
        cls.networks = NetworkConfig.erdosrenyi(1000, 4.0 / 999.0)
        cls.dynamics = DynamicsConfig.sis_default()
        cls.model = DynamicsConfig.sis_gnn_default()

        cls.train_details = TrainingConfig.test()
        cls.metrics = MetricsConfig.test()
        cls.summaries = SummariesConfig.test()
        cls.train_metrics = []
        cls.callbacks = CallbackConfig.default(cls.path_to_best)

        cls.seed = 0

        return cls
