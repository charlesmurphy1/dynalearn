from .base_metrics import Metrics
import numpy as np
import tqdm

class MeanfieldMetrics(Metrics):
        def __init__(self, p_k, verbose=1):
            super(MeanfieldMetrics, self).__init__(verbose)
            self.p_k = p_k
            return

    def compute(self, experiment):
        return
