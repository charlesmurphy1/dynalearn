import numpy as np

from dynalearn.datasets.transforms import StateTransform


class RemapStateTransform(StateTransform):
    def setup(self, experiment):
        self.state_map = experiment.dynamics.state_map

    def _transform_state_(self, x):
        _x = np.vectorize(self.state_map.get)(x.copy())
        return _x


class PartiallyRemapStateTransform(StateTransform):
    def setup(self, experiment):
        self.state_map = experiment.dynamics.state_map
        self.hide_prob = experiment.dynamics.hide_prob

    def _transform_state_(self, x):
        _x = np.vectorize(self.state_map.get)(x.copy())
        y = x.copy()
        n_remap = np.random.binomial(_x.shape[0], self.hide_prob)
        index = np.random.choice(range(_x.shape[0]), size=n_remap, replace=False)
        y[index] = _x[index]
        return y
