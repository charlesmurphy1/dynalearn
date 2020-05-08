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

    @abstractmethod
    def get_metrics(self, experiment):
        raise NotImplemented

    def initialize(self, experiment):
        self.metrics1, self.metrics2 = self.get_metrics(experiment)

        self.data["all"] = self._get_summary_("ltp")
        self.data["train"] = self._get_summary_("train_ltp")
        if "val_ltp" in self.metrics1.data and "val_ltp" in self.metrics2.data:
            self.data["val"] = self._get_summary_("val_ltp")
        if "test_ltp" in self.metrics1.data and "test_ltp" in self.metrics2.data:
            self.data["test"] = self._get_summary_("test_ltp")

    def _get_summary_(self, name):
        jsd = LTPMetrics.compare(
            self.metrics1.data[name],
            self.metrics2.data[name],
            self.metrics1.data["summaries"],
            func=jensenshannon,
        )
        x, y, el, eh = LTPMetrics.aggregate(
            jsd, self.metrics1.data["summaries"], err_reduce=self.err_reduce
        )
        return np.array([x, y, el, eh]).T


class TrueGNNJSDErrorSummary(JSDErrorSummary):
    def get_metrics(self, experiment):
        m1 = experiment.metrics["TrueLTPMetrics"]
        m2 = experiment.metrics["GNNLTPMetrics"]

        return m1, m2


class TrueMLEJSDErrorSummary(JSDErrorSummary):
    def get_metrics(self, experiment):
        m1 = experiment.metrics["TrueLTPMetrics"]
        m2 = experiment.metrics["MLELTPMetrics"]

        return m1, m2


class StarJSDErrorSummary(JSDErrorSummary):
    def initialize(self, experiment):
        self.metrics1, self.metrics2 = self.get_metrics(experiment)
        self.data["all"] = self._get_summary_("ltp")


class TrueGNNStarJSDErrorSummary(StarJSDErrorSummary):
    def get_metrics(self, experiment):
        m1 = experiment.metrics["TrueStarLTPMetrics"]
        m2 = experiment.metrics["GNNStarLTPMetrics"]
        return m1, m2


class TrueUniformStarJSDErrorSummary(StarJSDErrorSummary):
    def get_metrics(self, experiment):
        m1 = experiment.metrics["TrueStarLTPMetrics"]
        m2 = experiment.metrics["UniformStarLTPMetrics"]
        return m1, m2
