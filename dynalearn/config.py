from dynalearn.utilities import get_schedule
from dynalearn.utilities.metrics import model_entropy
from tensorflow.keras.optimizers import get
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.backend import variable
import numpy as np


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
        cls.step_per_epoch = 1000
        cls.val_fraction = 0.1
        cls.val_bias = 0.8
        cls.test_fraction = None
        cls.test_bias = 0.8
        cls.np_seed = 1

        np.random.seed(cls.np_seed)
        cls.optimizer = get(cls.name_optimizer)
        cls.optimizer.lr = variable(cls.initial_lr)
        cls.callbacks = [LearningRateScheduler(get_schedule(cls.schedule), verbose=1)]
        cls.training_metrics = [model_entropy]

        return cls

    def test(cls):

        cls = cls()

        cls.name_optimizer = "Adam"
        cls.initial_lr = 0.0005
        cls.schedule = {"epoch": 10, "factor": 2}
        cls.num_epochs = 5
        cls.num_graphs = 1
        cls.num_samples = 100
        cls.step_per_epoch = 1000
        cls.val_fraction = None
        cls.val_bias = 0.8
        cls.test_fraction = None
        cls.test_bias = 0.8
        cls.np_seed = 1

        np.random.seed(cls.np_seed)
        cls.optimizer = get(cls.name_optimizer)
        cls.optimizer.lr = variable(cls.initial_lr)
        cls.callbacks = [LearningRateScheduler(get_schedule(cls.schedule), verbose=1)]
        cls.training_metrics = [model_entropy]

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
        cls.step_per_epoch = 1000
        cls.val_fraction = 0.1
        cls.val_bias = 0.8
        cls.test_fraction = None
        cls.test_bias = 0.8
        cls.np_seed = 1

        np.random.seed(cls.np_seed)
        cls.optimizer = get(cls.name_optimizer)
        cls.optimizer.lr = variable(cls.initial_lr)
        cls.callbacks = [LearningRateScheduler(get_schedule(cls.schedule), verbose=1)]
        cls.training_metrics = [model_entropy]

        return cls
