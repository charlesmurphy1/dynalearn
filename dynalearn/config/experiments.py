import dynalearn as dl
import os

from dynalearn.config import *


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
    def sis_er(
        cls,
        name=None,
        path_to_data="./",
        path_to_best="./",
        path_to_summary="./",
        mode="fast",
    ):
        cls = cls()
        if name is None:
            cls.name = "sis-er"
        else:
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
        cls.networks = NetworkConfig.er_default()
        cls.dynamics = DynamicsConfig.sis_default()
        cls.model = DynamicsConfig.sis_gnn_default()

        cls.train_details = TrainingConfig.default()
        if mode == "fast":
            cls.post_metrics = MetricsConfig.sis_fast()
            cls.summaries = SummariesConfig.sis_fast()
        elif mode == "complete":
            cls.post_metrics = MetricsConfig.sis_complete()
            cls.summaries = SummariesConfig.sis_complete()
        else:
            raise ValueError(
                f"{mode} is invalid, valid entries are ['fast', 'complete']."
            )
        cls.train_metrics = ["jensenshannon", "model_entropy"]
        cls.callbacks = CallbackConfig.default(cls.path_to_best)

        cls.seed = 0

        return cls

    @classmethod
    def sis_ba(
        cls,
        name=None,
        path_to_data="./",
        path_to_best="./",
        path_to_summary="./",
        mode="fast",
    ):
        cls = cls()
        if name is None:
            cls.name = "sis-ba"
        else:
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
        cls.networks = NetworkConfig.ba_default()
        cls.dynamics = DynamicsConfig.sis_default()
        cls.model = DynamicsConfig.sis_gnn_default()

        cls.train_details = TrainingConfig.default()
        if mode == "fast":
            cls.post_metrics = MetricsConfig.sis_fast()
            cls.summaries = SummariesConfig.sis_fast()
        elif mode == "complete":
            cls.post_metrics = MetricsConfig.sis_complete()
            cls.summaries = SummariesConfig.sis_complete()
        else:
            raise ValueError(
                f"{mode} is invalid, valid entries are ['fast', 'complete']."
            )
        cls.train_metrics = ["jensenshannon", "model_entropy"]
        cls.callbacks = CallbackConfig.default(cls.path_to_best)

        cls.seed = 0

        return cls

    @classmethod
    def plancksis_er(
        cls,
        name=None,
        path_to_data="./",
        path_to_best="./",
        path_to_summary="./",
        mode="fast",
    ):
        cls = cls()
        if name is None:
            cls.name = "plancksis-er"
        else:
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
        cls.networks = NetworkConfig.er_default()
        cls.dynamics = DynamicsConfig.plancksis_default()
        cls.model = DynamicsConfig.plancksis_gnn_default()

        cls.train_details = TrainingConfig.default()
        if mode == "fast":
            cls.post_metrics = MetricsConfig.plancksis_fast()
            cls.summaries = SummariesConfig.plancksis_fast()
        elif mode == "complete":
            cls.post_metrics = MetricsConfig.plancksis_complete()
            cls.summaries = SummariesConfig.plancksis_complete()
        else:
            raise ValueError(
                f"{mode} is invalid, valid entries are ['fast', 'complete']."
            )
        cls.train_metrics = ["jensenshannon", "model_entropy"]
        cls.callbacks = CallbackConfig.default(cls.path_to_best)

        cls.seed = 0

        return cls

    @classmethod
    def plancksis_ba(
        cls,
        name=None,
        path_to_data="./",
        path_to_best="./",
        path_to_summary="./",
        mode="fast",
    ):
        cls = cls()
        if name is None:
            cls.name = "plancksis-ba"
        else:
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
        cls.networks = NetworkConfig.ba_default()
        cls.dynamics = DynamicsConfig.plancksis_default()
        cls.model = DynamicsConfig.plancksis_gnn_default()

        cls.train_details = TrainingConfig.default()
        if mode == "fast":
            cls.post_metrics = MetricsConfig.plancksis_fast()
            cls.summaries = SummariesConfig.plancksis_fast()
        elif mode == "complete":
            cls.post_metrics = MetricsConfig.plancksis_complete()
            cls.summaries = SummariesConfig.plancksis_complete()
        else:
            raise ValueError(
                f"{mode} is invalid, valid entries are ['fast', 'complete']."
            )
        cls.train_metrics = ["jensenshannon", "model_entropy"]
        cls.callbacks = CallbackConfig.default(cls.path_to_best)

        cls.seed = 0

        return cls

    @classmethod
    def sissis_er(
        cls,
        name=None,
        path_to_data="./",
        path_to_best="./",
        path_to_summary="./",
        mode="fast",
    ):
        cls = cls()
        if name is None:
            cls.name = "sissis-er"
        else:
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
        cls.networks = NetworkConfig.er_default()
        cls.dynamics = DynamicsConfig.sissis_default()
        cls.model = DynamicsConfig.sissis_gnn_default()

        cls.train_details = TrainingConfig.default()
        if mode == "fast":
            cls.post_metrics = MetricsConfig.sissis_fast()
            cls.summaries = SummariesConfig.sissis_fast()
        elif mode == "complete":
            cls.post_metrics = MetricsConfig.sissis_complete()
            cls.summaries = SummariesConfig.sissis_complete()
        else:
            raise ValueError(
                f"{mode} is invalid, valid entries are ['fast', 'complete']."
            )
        cls.train_metrics = ["jensenshannon", "model_entropy"]
        cls.callbacks = CallbackConfig.default(cls.path_to_best)

        cls.seed = 0

        return cls

    @classmethod
    def sissis_ba(
        cls,
        name=None,
        path_to_data="./",
        path_to_best="./",
        path_to_summary="./",
        mode="fast",
    ):
        cls = cls()
        if name is None:
            cls.name = "sissis-ba"
        else:
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
        cls.networks = NetworkConfig.ba_default()
        cls.dynamics = DynamicsConfig.sissis_default()
        cls.model = DynamicsConfig.sissis_gnn_default()

        cls.train_details = TrainingConfig.default()
        if mode == "fast":
            cls.post_metrics = MetricsConfig.sissis_fast()
            cls.summaries = SummariesConfig.sissis_fast()
        elif mode == "complete":
            cls.post_metrics = MetricsConfig.sissis_complete()
            cls.summaries = SummariesConfig.sissis_complete()
        else:
            raise ValueError(
                f"{mode} is invalid, valid entries are ['fast', 'complete']."
            )
        cls.train_metrics = ["jensenshannon", "model_entropy"]
        cls.callbacks = CallbackConfig.default(cls.path_to_best)

        cls.seed = 0

        return cls

    @classmethod
    def test(cls):
        cls = cls()
        path = "/home/charles/Documents/ulaval/doctorat/projects/dynalearn-all/dynalearn/data/test"
        cls.name = "test"
        cls.path_to_data = os.path.join(path, cls.name)
        if not os.path.exists(cls.path_to_data):
            os.makedirs(cls.path_to_data)
        cls.path_to_best = os.path.join(path, "best", cls.name + ".pt")
        if not os.path.exists(os.path.join(path, "best")):
            os.makedirs(os.path.join(path, "best"))
        cls.dataset = DatasetConfig.state_weighted_default()
        cls.networks = NetworkConfig.erdosrenyi(100, 4.0 / 99.0)
        cls.dynamics = DynamicsConfig.sis_default()
        cls.model = DynamicsConfig.gnn_test()

        cls.train_details = TrainingConfig.test()
        cls.post_metrics = MetricsConfig.test()
        cls.train_metrics = []
        cls.callbacks = CallbackConfig.default(cls.path_to_best)

        cls.seed = 0

        return cls
