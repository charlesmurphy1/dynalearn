from dynalearn.utilities import get_schedule
from tensorflow.keras.optimizers import get
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.backend import variable


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
        cls.val_fraction = 0.01
        cls.val_bias = 1
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
        cls.val_fraction = 0.01
        cls.val_bias = 0.8
        cls.test_fraction = None
        cls.test_bias = 0.8
        cls.np_seed = 1
        cls.training_metrics = ["model_entropy", "jensenshannon"]

        return cls
