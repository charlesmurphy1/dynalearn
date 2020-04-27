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

        self.get_data["all"] = lambda: self._get_summary_("ltp")
        self.names.append("all")
        self.get_data["train"] = lambda: self._get_summary_("train_ltp")
        self.names.append("train")
        if "val_ltp" in self.metrics1.data and "val_ltp" in self.metrics2.data:
            self.get_data["val"] = lambda: self._get_summary_("val_ltp")
            self.names.append("val")
        if "test_ltp" in self.metrics1.data and "test_ltp" in self.metrics2.data:
            self.get_data["test"] = lambda: self._get_summary_("test_ltp")
            self.names.append("test")

    def _get_summary_(self, name):
        jsd = LTPMetrics.compare(
            self.metrics1.data[name],
            self.metrics1.data[name],
            self.metrics1.data["summaries"],
            func=jensenshannon,
        )
        x, y, el, eh = LTPMetrics.aggregate(
            jsd, self.metrics1.data["summaries"], err_reduce=self.err_reduce
        )
        return np.array([x, y, el, eh]).T


class TrueGNNJSDErrorSummary(JSDErrorSummary):
    def get_metrics(self, experiment):
        m1 = experiment.post_metrics["TrueLTPMetrics"]
        m2 = experiment.post_metrics["GNNLTPMetrics"]

        return m1, m2


class TrueMLEJSDErrorSummary(JSDErrorSummary):
    def get_metrics(self, experiment):
        m1 = experiment.post_metrics["TrueLTPMetrics"]
        m2 = experiment.post_metrics["MLELTPMetrics"]

        return m1, m2


class StarJSDErrorSummary(JSDErrorSummary):
    def initialize(self, experiment):
        self.metrics1, self.metrics2 = self.get_metrics(experiment)

        self.get_data["all"] = lambda: self._get_summary_("ltp")
        self.names.append(f"all")


class TrueGNNStarJSDErrorSummary(StarJSDErrorSummary):
    def get_metrics(self, experiment):
        m1 = experiment.post_metrics["TrueStarLTPMetrics"]
        m2 = experiment.post_metrics["GNNStarLTPMetrics"]
        return m1, m2


class TrueUniformStarJSDErrorSummary(StarJSDErrorSummary):
    def get_metrics(self, experiment):
        m1 = experiment.post_metrics["TrueStarLTPMetrics"]
        m2 = experiment.post_metrics["UniformStarLTPMetrics"]
        return m1, m2
