import numpy as np


class Initializer:
    def __init__(self, config):
        self.config = config
        self.init_param = config.init_param
        self.adaptive = config.adaptive
        self.all_modes = list(self.init_param.keys())
        self.num_modes = len(self.init_param.keys())
        self._mode = self.all_modes[0]

    def __call__(self):
        _x0 = self.dynamics.initial_state(init_param=self.init_param[self.mode])
        x0 = np.zeros((*_x0.shape, self.window_size * self.window_step))
        x0.T[0] = _x0.T
        for i in range(1, self.window_size * self.window_step):
            x0.T[i] = self.dynamics.sample(x0[i - 1]).T
        return x0

    def setUp(self, metrics):
        self.dynamics = metrics.dynamics
        self.num_states = metrics.model.num_states
        self.window_size = metrics.model.window_size
        self.window_step = metrics.model.window_step

    def update(self, x):
        assert x.shape == (self.num_states,)
        if self.adaptive:
            self.init_param[self.mode] = x

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode in self.all_modes:
            self._mode = mode
        else:
            raise ValueError(
                f"Mode `{mode}` is not invalid, available modes are `{self.all_modes}`"
            )