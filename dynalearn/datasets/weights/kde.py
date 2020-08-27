import numpy as np
from scipy.stats import gaussian_kde


class KernelDensityEstimator:
    def __init__(self, samples, max_num_samples=-1):
        assert isinstance(samples, list)
        self.samples = samples
        if isinstance(samples[0], np.ndarray):
            self.shape = samples[0].shape
        elif isinstance(samples[0], (int, float)):
            self.shape = (1,)
        for s in samples:
            if isinstance(s, (int, float)):
                s = np.array([s])
            assert s.shape == self.shape
        self.max_num_samples = max_num_samples
        self.kde = None
        self._mean = None
        self._std = None
        self._norm = None
        self._index = None
        self.get_kde()

    def pdf(self, x):
        if isinstance(x, list):
            x = np.array(x)
            x = x.reshape(x.shape[0], -1).T
        if x.shape == self.shape:
            x = np.expand_dims(x, -1)
        assert x.shape[:-1] == self.shape

        if self.kde is None:
            return np.ones(x.shape[-1]) / self._norm
        else:
            y = (x[self._index] - self._mean[self._index]) / self._std[self._index]
            p = self.kde.pdf(y) / self._norm
            return p

    def get_kde(self):
        if len(self.samples) <= 1:
            self._norm = 1
            return
        x = np.array(self.samples)
        x = x.reshape(x.shape[0], -1).T
        mean = np.expand_dims(x.mean(axis=-1), -1)
        std = np.expand_dims(x.std(axis=-1), -1)
        if np.all(std < 1e-8):
            self._norm = len(self.samples)
            return
        self._index = np.where(std > 1e-8)[0]
        y = (x[self._index] - mean[self._index]) / std[self._index]
        if self.max_num_samples < y.shape[-1] and self.max_num_samples != -1:
            ind = np.random.choice(range(y.shape[-1]), size=self.max_num_samples)
            y = y[:, ind]
        self.kde = gaussian_kde(y)
        self._mean = mean
        self._std = std
        self._norm = self.kde.pdf(y).sum()
        self.samples = []
