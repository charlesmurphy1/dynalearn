from dynalearn.utilities import get_schedule
from tensorflow.keras.optimizers import get
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.backend import variable
import getpass
import pickle
import os
import dynalearn as dl


class TrainingConfig:
    @classmethod
    def default(cls):

        cls = cls()

        cls.name_optimizer = "Adam"
        cls.initial_lr = 0.0005
        cls.schedule = {"epoch": 10, "factor": 2}
        cls.num_epochs = 20
        cls.num_graphs = 1
        cls.num_samples = 10000
        cls.step_per_epoch = 10000
        cls.sampling_bias = 0.6
        cls.val_fraction = 0.01
        cls.val_bias = 0.8
        cls.test_fraction = None
        cls.test_bias = 0.8
        cls.np_seed = 1
        cls.training_metrics = ["model_entropy", "jensenshannon"]

        return cls

    @classmethod
    def test(cls):

        cls = cls()

        cls.name_optimizer = "Adam"
        cls.initial_lr = 0.0005
        cls.schedule = {"epoch": 10, "factor": 2}
        cls.num_epochs = 5
        cls.num_graphs = 1
        cls.num_samples = 100
        cls.step_per_epoch = 100
        cls.sampling_bias = 0.6
        cls.val_fraction = 0.01
        cls.val_bias = 0.8
        cls.test_fraction = None
        cls.test_bias = 0.8
        cls.np_seed = 1
        cls.training_metrics = ["model_entropy", "jensenshannon"]

        return cls

    @classmethod
    def changing_num_samples(cls, num_samples):

        cls = cls()

        cls.name_optimizer = "Adam"
        cls.initial_lr = 0.0005
        cls.schedule = {"epoch": 10, "factor": 2}
        cls.num_epochs = 20
        cls.num_graphs = 1
        cls.num_samples = num_samples
        cls.step_per_epoch = 10000
        cls.sampling_bias = 0.6
        cls.val_fraction = 0.01
        cls.val_bias = 0.6
        cls.test_fraction = None
        cls.test_bias = 0.8
        cls.np_seed = 1
        # cls.training_metrics = ["model_entropy"]
        cls.training_metrics = ["model_entropy", "jensenshannon"]

        return cls


class ExperimentConfig:
    def __init__(self):
        self.config = {}
        self.path_to_config = None

    def set_config(
        self,
        name,
        num_samples,
        dynamics,
        graph,
        model,
        metrics,
        path_to_dir,
        path_to_model,
    ):
        training_config = TrainingConfig.changing_num_samples(num_samples)
        self.config["name"] = name
        self.config["dynamics"] = dynamics
        self.config["graph"] = graph
        self.config["model"] = model
        self.config["generator"] = {
            "name": "DynamicsGenerator",
            "config": dl.datasets.GeneratorConfig.default(),
            "sampler": {
                "name": "StateBiasedSampler",
                "config": dl.datasets.samplers.SamplerConfig.BiasedSamplerDefault(
                    dynamics, training_config.sampling_bias
                ),
            },
        }
        self.config["metrics"] = {
            "name": [
                "AttentionMetrics",
                "TrueLTPMetrics",
                "GNNLTPMetrics",
                "MLELTPMetrics",
                "TrueStarLTPMetrics",
                "GNNStarLTPMetrics",
                "UniformStarLTPMetrics",
                "StatisticsMetrics",
                "TruePEMFMetrics",
                "GNNPEMFMetrics",
                "TruePESSMetrics",
                "GNNPESSMetrics",
            ],
            "config": metrics,
        }
        self.config["training"] = training_config
        self.config["path_to_dir"] = path_to_dir
        self.config["path_to_bestmodel"] = path_to_model

    def save(self, path=None, overwrite=True):
        if path is None:
            path = os.path.join(self.config["path_to_dir"], "config.pickle")
        else:
            path = os.path.join(path, "config.pickle")

        self.path_to_config = path
        if os.path.exists(path) and not overwrite:
            return

        with open(path, "wb") as f:
            pickle.dump(self.config, f)

    @classmethod
    def config_from_file(cls, filename):
        cls = cls()
        with open(filename, "rb") as f:
            cls.config = pickle.load(f)

        return cls

    @classmethod
    def sis_ba(
        cls, num_samples=10000, path_to_dir="../data/", path_to_model="../data/"
    ):
        cls = cls()
        name = "sis-ba-{0}".format(num_samples)
        dynamics = {
            "name": "SIS",
            "params": {"infection": 0.04, "recovery": 0.08, "init": "None"},
        }
        graph = {"name": "BAGraph", "params": {"N": 1000, "M": 2}}
        model = {"name": "EpidemicPredictor", "config": dl.models.GNNConfig.SISGNN()}
        metrics = dl.metrics.MetricsConfig.SISMetrics()
        cls.set_config(
            name,
            num_samples,
            dynamics,
            graph,
            model,
            metrics,
            path_to_dir,
            path_to_model,
        )
        return cls

    @classmethod
    def sis_er(
        cls, num_samples=10000, path_to_dir="../data/", path_to_model="../data/"
    ):
        cls = cls()
        name = "sis-er-{0}".format(num_samples)
        dynamics = {
            "name": "SIS",
            "params": {"infection": 0.04, "recovery": 0.08, "init": "None"},
        }
        graph = {"name": "ERGraph", "params": {"N": 1000, "density": 0.004}}
        model = {"name": "EpidemicPredictor", "config": dl.models.GNNConfig.SISGNN()}
        metrics = dl.metrics.MetricsConfig.SISMetrics()
        cls.set_config(
            name,
            num_samples,
            dynamics,
            graph,
            model,
            metrics,
            path_to_dir,
            path_to_model,
        )
        return cls

    @classmethod
    def plancksis_ba(
        cls, num_samples=10000, path_to_dir="../data/", path_to_model="../data/"
    ):
        cls = cls()
        name = "plancksis-ba-{0}".format(num_samples)
        dynamics = {
            "name": "PlanckSIS",
            "params": {"temperature": 10, "recovery": 0.07, "init": "None"},
        }
        graph = {"name": "BAGraph", "params": {"N": 1000, "M": 2}}
        model = {
            "name": "EpidemicPredictor",
            "config": dl.models.GNNConfig.ComplexSISGNN(),
        }
        metrics = dl.metrics.MetricsConfig.PlanckSISMetrics()
        cls.set_config(
            name,
            num_samples,
            dynamics,
            graph,
            model,
            metrics,
            path_to_dir,
            path_to_model,
        )
        return cls

    @classmethod
    def plancksis_er(
        cls, num_samples=10000, path_to_dir="../data/", path_to_model="../data/"
    ):
        cls = cls()
        name = "plancksis-er-{0}".format(num_samples)
        dynamics = {
            "name": "PlanckSIS",
            "params": {"temperature": 10, "recovery": 0.07, "init": "None"},
        }
        graph = {"name": "ERGraph", "params": {"N": 1000, "density": 0.004}}
        model = {
            "name": "EpidemicPredictor",
            "config": dl.models.GNNConfig.ComplexSISGNN(),
        }
        metrics = dl.metrics.MetricsConfig.PlanckSISMetrics()
        cls.set_config(
            name,
            num_samples,
            dynamics,
            graph,
            model,
            metrics,
            path_to_dir,
            path_to_model,
        )
        return cls

    @classmethod
    def sissis_ba(
        cls, num_samples=10000, path_to_dir="../data/", path_to_model="../data/"
    ):
        cls = cls()
        name = "sissis-ba-{0}".format(num_samples)
        dynamics = {
            "name": "SISSIS",
            "params": {
                "infection1": 0.02,
                "infection2": 0.01,
                "recovery1": 0.12,
                "recovery2": 0.13,
                "coupling": 10,
                "init": "None",
            },
        }
        graph = {"name": "BAGraph", "params": {"N": 1000, "M": 2}}
        model = {"name": "EpidemicPredictor", "config": dl.models.GNNConfig.SISSISGNN()}
        metrics = dl.metrics.MetricsConfig.SISSISMetrics()
        cls.set_config(
            name,
            num_samples,
            dynamics,
            graph,
            model,
            metrics,
            path_to_dir,
            path_to_model,
        )
        return cls

    @classmethod
    def sissis_er(
        cls, num_samples=10000, path_to_dir="../data/", path_to_model="../data/"
    ):
        cls = cls()
        name = "sissis-er-{0}".format(num_samples)
        dynamics = {
            "name": "SISSIS",
            "params": {
                "infection1": 0.02,
                "infection2": 0.01,
                "recovery1": 0.12,
                "recovery2": 0.13,
                "coupling": 10,
                "init": "None",
            },
        }
        graph = {"name": "ERGraph", "params": {"N": 1000, "density": 0.004}}
        model = {"name": "EpidemicPredictor", "config": dl.models.GNNConfig.SISSISGNN()}
        metrics = dl.metrics.MetricsConfig.SISSISMetrics()
        cls.set_config(
            name,
            num_samples,
            dynamics,
            graph,
            model,
            metrics,
            path_to_dir,
            path_to_model,
        )
        return cls
