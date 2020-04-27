import h5py
import tqdm

from abc import abstractmethod
from dynalearn.experiments.metrics import Metrics


class Summary(Metrics):
    def compute(self, experiment, verbose=None):
        self.verbose = verbose or self.verbose
        self.initialize(experiment)

        for k in self.names:
            self.data[k] = self.get_data[k]()
