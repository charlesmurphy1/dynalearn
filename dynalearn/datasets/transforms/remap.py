import numpy as np

from dynalearn.datasets.transforms import StateTransform


class RemapStateTransform(StateTransform):
    def setup(self, experiment):
        self.state_map = experiment.dynamics.state_map

    def _transform_state_(self, x):
        return np.vectorize(self.state_map.get)(x)
