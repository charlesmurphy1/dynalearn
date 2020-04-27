import numpy as np

from dynalearn.experiments.summaries import Summary
from dynalearn.experiments.metrics import LTPMetrics
from abc import abstractmethod


class LTPSummary(Summary):
    def __init__(self, config, verbose=0):
        Summary.__init__(self, config, verbose)
        self.transitions = config.transitions
        self.axis = config.axis
        if "err_reduce" not in config.__dict__:
            self.err_reduce = "percentile"
        else:
            self.err_reduce = config.err_reduce

    @abstractmethod
    def get_metrics(self, experiment):
        raise NotImplemented

    def initialize(self, experiment):
        self.metrics = self.get_metrics(experiment)

        for u, v in self.transitions:
            self.get_data[f"all/{u}-{v}"] = lambda: self._get_summary_("ltp", u, v)
            self.names.append(f"all/{u}-{v}")
            self.get_data[f"train/{u}-{v}"] = lambda: self._get_summary_(
                "train_ltp", u, v
            )
            self.names.append(f"train/{u}-{v}")
            if "val_ltp" in self.metrics.data:
                self.get_data[f"val/{u}-{v}"] = lambda: self._get_summary_(
                    "val_ltp", u, v
                )
                self.names.append(f"val/{u}-{v}")
            if "test_ltp" in self.metrics.data:
                self.get_data[f"test/{u}-{v}"] = lambda: self._get_summary_(
                    "test_ltp", u, v
                )
                self.names.append(f"test/{u}-{v}")

    def _get_summary_(self, name, in_s, out_s):
        summaries = self.metrics.summaries
        x, y, yl, yh = LTPMetrics.aggregate(
            self.metrics.data[name],
            self.metrics.data["summaries"],
            in_state=in_s,
            out_state=out_s,
            axis=self.axis,
            reduce="mean",
            err_reduce=self.err_reduce,
        )
        return np.array([x, y, yl, yh]).T


class TrueLTPSummary(LTPSummary):
    def get_metrics(self, experiment):
        return experiment.post_metrics["TrueLTPMetrics"]


class GNNLTPSummary(LTPSummary):
    def get_metrics(self, experiment):
        return experiment.post_metrics["GNNLTPMetrics"]


class MLELTPSummary(LTPSummary):
    def get_metrics(self, experiment):
        return experiment.post_metrics["MLELTPMetrics"]


class StarLTPSummary(LTPSummary):
    def initialize(self, experiment):
        self.metrics = self.get_metrics(experiment)

        for u, v in self.transitions:
            self.get_data[f"{u}-{v}"] = lambda: self._get_summary_("ltp", u, v)
            self.names.append(f"{u}-{v}")


class TrueStarLTPSummary(StarLTPSummary):
    def get_metrics(self, experiment):
        return experiment.post_metrics["TrueStarLTPMetrics"]


class GNNStarLTPSummary(StarLTPSummary):
    def get_metrics(self, experiment):
        return experiment.post_metrics["GNNStarLTPMetrics"]


class UniformStarLTPSummary(StarLTPSummary):
    def get_metrics(self, experiment):
        return experiment.post_metrics["UniformStarLTPMetrics"]
