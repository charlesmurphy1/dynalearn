import numpy as np

from abc import abstractmethod, ABC
from dynalearn.utilities import from_nary


class ModelSampler(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def __call__(self, mode, initializer, statistics):
        raise NotImplemented

    @abstractmethod
    def aggregate(self, x):
        raise NotImplemented

    def setUp(self, metrics):
        self.dynamics = metrics.dynamics
        self.num_states = metrics.model.num_states
        self.window_size = metrics.model.window_size
        self.window_step = metrics.model.window_step

    def burning(self, model, x, burn=1):
        for b in range(burn):
            y = x.T[:: self.window_step]
            y = model.sample(y.T)
            x = np.roll(x, -1, axis=-1)
            x.T[-1] = y.T
        return x

    @classmethod
    def getter(cls, config):
        __all_samplers__ = {
            "SteadyStateSampler": SteadyStateSampler,
            "FixedPointSampler": FixedPointSampler,
        }

        if config.sampler in __all_samplers__:
            return __all_samplers__[config.sampler](config)
        else:
            raise ValueError(
                f"`{config.sampler}` is invalid, valid entries are `{__all_samplers__.key()}`"
            )


class SteadyStateSampler(ModelSampler):
    def __init__(self, config):
        ModelSampler.__init__(self, config)
        self.initial_burn = config.initial_burn
        self.num_windows = config.num_windows
        self.mid_burn = config.mid_burn

    def __call__(self, model, initializer):
        x0 = initializer()
        x0 = self.burning(model, x0, self.initial_burn)
        samples = []
        for i in range(self.num_windows):
            x0 = self.burning(model, x0, self.mid_burn)
            samples.append(self.aggregate(x0))
            if self.dynamics.is_dead(x0):
                x0 = initializer()
        return samples

    def aggregate(self, x):
        agg_x = []
        assert x.shape[-1] == self.window_size * self.window_step
        if x.ndim == 2:
            x = from_nary(x[:, :: self.window_step], axis=-1, base=self.num_states)
        elif x.ndim == 3:
            np.array(
                [
                    from_nary(xx[:, :: self.window_step], axis=-1, base=self.num_states)
                    for xx in x
                ]
            )
        for i in range(self.num_states ** self.window_size):
            agg_x.append(np.mean(x == i, axis=0))
        return np.array(agg_x)


class FixedPointSampler(ModelSampler):
    def __init__(self, config):
        ModelSampler.__init__(self, config)
        self.initial_burn = config.initial_burn
        self.mid_burn = config.mid_burn
        self.tol = config.tol
        self.maxiter = config.maxiter

    def __call__(self, model, initializer):
        x = initializer()
        x = self.burning(model, x, self.initial_burn)

        diff = np.inf
        i = 0
        while diff > self.tol and i < self.maxiter:
            y = self.burning(model, x, self.mid_burn)
            diff = self.distance(x, y)
            x = y * 1
            i += 1
        y = self.aggregate(y)
        return y

    def distance(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def aggregate(self, x):
        assert x.shape[-1] == self.window_size * self.window_step
        assert x.shape[-2] == self.num_states
        return x.mean((0, -1))
