import numpy as np

from dynalearn.experiments.summaries import Summary
from dynalearn.experiments.metrics import LTPMetrics
from abc import abstractmethod
from scipy.spatial.distance import jensenshannon


class JSDErrorSummary(Summary):
    def __init__(self, config, verbose=0):
        Summary.__init__(self, config, verbose)
        if "err_reduce" not in config.__dict__:
            self.err_reduce = "percentile"
        else:
            self.err_reduce = config.err_reduce

    def get_metrics(self, experiment):
        ref_metrics = experiment.metrics["TrueLTPMetrics"]
        metrics = {}
        if "GNNLTPMetrics" in experiment.metrics:
            metrics["true-gnn"] = experiment.metrics["GNNLTPMetrics"]
        if "MLELTPMetrics" in experiment.metrics:
            metrics["true-mle"] = experiment.metrics["MLELTPMetrics"]
        if "UniformLTPMetrics" in experiment.metrics:
            metrics["true-uni"] = experiment.metrics["UniformLTPMetrics"]
        return ref_metrics, metrics

    def initialize(self, experiment):
        self.ref_metrics, self.metrics = self.get_metrics(experiment)
        data_all = self._get_summary_("ltp")
        data_train = self._get_summary_("train_ltp")
        for k in data_all:
            self.data[f"all-{k}"] = data_all[k]
            self.data[f"train-{k}"] = data_train[k]
        if "val_ltp" in self.ref_metrics.data:
            data_val = self._get_summary_("val_ltp")
            for k in data_val:
                self.data[f"val-{k}"] = data_val[k]
        if "test_ltp" in self.ref_metrics.data:
            data_test = self._get_summary_("test_ltp")
            for k in data_test:
                self.data[f"test-{k}"] = data_test[k]

    def _get_summary_(self, name):
        data = {}
        for k, m in self.metrics.items():
            jsd = LTPMetrics.compare(
                self.ref_metrics.data[name],
                m.data[name],
                self.ref_metrics.data["summaries"],
                func=jensenshannon,
            )
            x, y, el, eh = LTPMetrics.aggregate(
                jsd, self.ref_metrics.data["summaries"], err_reduce=self.err_reduce
            )
            data[k] = np.array([x, y, el, eh]).T
        return data


class StarJSDErrorSummary(JSDErrorSummary):
    def initialize(self, experiment):
        self.ref_metrics, self.metrics = self.get_metrics(experiment)
        data = self._get_summary_("ltp")
        for k in data:
            self.data[f"all-{k}"] = data[k]

    def get_metrics(self, experiment):
        ref_metrics = experiment.metrics["TrueStarLTPMetrics"]
        metrics = {}
        if "GNNStarLTPMetrics" in experiment.metrics:
            metrics["true-gnn"] = experiment.metrics["GNNStarLTPMetrics"]
        if "MLEStarLTPMetrics" in experiment.metrics:
            metrics["true-mle"] = experiment.metrics["MLEStarLTPMetrics"]
        if "UniformStarLTPMetrics" in experiment.metrics:
            metrics["true-uni"] = experiment.metrics["UniformStarLTPMetrics"]
        return ref_metrics, metrics
