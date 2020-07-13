import networkx as nx
import numpy as np

from .ltp import LTPMetrics, TrueLTPMetrics, GNNLTPMetrics, UniformLTPMetrics
from dynalearn.utilities import all_combinations, from_nary, to_nary
from scipy.special import binom
from itertools import product


class StarLTPMetrics(LTPMetrics):
    def __init__(self, config, verbose=0):
        LTPMetrics.__init__(self, config, verbose)
        self.degree_class = config.degree_class
        self.names = ["ltp", "summaries"]

    def initialize(self, experiment):
        self.model = self.get_model(experiment)
        eff_num_states = self.num_states ** self.window_size
        self.num_updates = np.sum(
            binom(self.degree_class + eff_num_states - 1, eff_num_states - 1)
            * eff_num_states
        ).astype("int")

        self.get_data["ltp"] = lambda pb: self._get_ltp_(pb=pb)
        self.get_data["summaries"] = lambda pb: self._get_summaries_(pb=pb)

    def _get_summaries_(self, pb=None):
        return np.array(list(self.summaries))

    def _get_ltp_(self, pb=None):
        eff_num_states = self.num_states ** self.window_size
        num_nodes = np.max(self.degree_class) + 1
        ltp = np.zeros((self.num_updates, self.num_states))
        i = 0
        for k in self.degree_class:
            g = nx.empty_graph(num_nodes)
            g.add_edges_from(nx.star_graph(k).edges())
            self.model.network = g
            all_s = range(eff_num_states)
            all_ns = all_combinations(k, eff_num_states)

            for s, ns in product(all_s, all_ns):
                inputs = np.zeros(num_nodes)
                inputs[0] = s
                inputs[1 : k + 1] = np.concatenate(
                    [j * np.ones(l) for j, l in enumerate(ns)]
                )
                inputs = to_nary(inputs, base=self.num_states, dim=self.window_size)
                ltp[i] = self.predict(inputs, inputs, None, None)[0]
                self.summaries.add((s, *ns))
                i += 1
                if pb is not None:
                    pb.update()
        return ltp


class TrueStarLTPMetrics(TrueLTPMetrics, StarLTPMetrics):
    def __init__(self, config, verbose=0):
        TrueLTPMetrics.__init__(self, config, verbose)
        StarLTPMetrics.__init__(self, config, verbose)


class GNNStarLTPMetrics(GNNLTPMetrics, StarLTPMetrics):
    def __init__(self, config, verbose=0):
        GNNLTPMetrics.__init__(self, config, verbose)
        StarLTPMetrics.__init__(self, config, verbose)


class UniformStarLTPMetrics(UniformLTPMetrics, StarLTPMetrics):
    def __init__(self, config, verbose=0):
        UniformLTPMetrics.__init__(self, config, verbose)
        StarLTPMetrics.__init__(self, config, verbose)
