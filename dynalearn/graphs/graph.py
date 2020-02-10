import networkx as nx
import numpy as np
import h5py
from random import choice


class GraphGenerator:
    def __init__(self, generator, params=dict()):
        self.generator = generator
        self.params = params
        self.instance_index = 0
        if "N" in params:
            self.num_nodes = params["N"]
        else:
            self.num_nodes = 1

    def generate(self):
        index = self.instance_index
        self.instance_index += 1
        name = type(self).__name__ + "_" + str(index)
        graph = self.generator(self.num_nodes)
        return name, graph


class CycleGraph(GraphGenerator):
    def __init__(self, params):
        generator = lambda N: nx.cycle_graph(N)
        super(CycleGraph, self).__init__(generator, params)


class CompleteGraph(GraphGenerator):
    def __init__(self, params):
        generator = lambda seed: nx.complete_graph(N)
        super(CompleteGraph, self).__init__(generator, params)


class StarGraph(GraphGenerator):
    def __init__(self, params):
        generator = lambda N: nx.star_graph(N - 1)
        super(StarGraph, self).__init__(generator, params)


class EmptyGraph(GraphGenerator):
    def __init__(self, params):
        generator = lambda N: nx.empty_graph(N)
        super(EmptyGraph, self).__init__(generator, params)


class RegularGraph(GraphGenerator):
    def __init__(self, params):
        generator = lambda N: nx.random_regular_graph(
            N, self.params["degree"], seed=np.random.randint(2 ** 31)
        )

        super(RegularGraph, self).__init__(generator, params)


class BAGraph(GraphGenerator):
    def __init__(self, params):
        generator = lambda N: nx.barabasi_albert_graph(
            N, self.params["M"], seed=np.random.randint(2 ** 31)
        )
        super(BAGraph, self).__init__(generator, params)


class ERGraph(GraphGenerator):
    def __init__(self, params):
        generator = lambda N: nx.gnp_random_graph(
            N, self.params["density"], seed=np.random.randint(2 ** 31)
        )
        super(ERGraph, self).__init__(generator, params)


class DegreeSequenceGraph(GraphGenerator):
    def __init__(self, params):
        self.degree_dist = params["degree_dist"]
        if "maxiter" not in params:
            self.maxiter = 100
        else:
            self.maxiter = params["maxiter"]
        super(DegreeSequenceGraph, self).__init__(self.__generator, params)

    def __generator(self, N):
        it = 0
        while it < self.maxiter:
            seq = self.degree_dist.sample(N)
            if np.sum(seq) % 2 == 0:
                return nx.configuration_model(seq, seed=np.random.randint(2 ** 31))
            it += 1

        raise ValueError("Invalid degree sequence.")


class PartyMix(GraphGenerator):
    def __init__(self, generators):
        self.generators = generators
        generator = lambda N: self._sample_generator()
        super(PartyMix, self).__init__(generator, params)
        self.params = dict()
        for g in self.generators:
            self.params.update(g.params)

    def _sample_generator(self):
        gen = choice(self.generators)
        n, g = gen.generate()
        return g
