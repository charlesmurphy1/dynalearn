import networkx as nx
import numpy as np
import h5py
from random import choice


class GraphGenerator:
    def __init__(self, generator):
        self.generator = generator
        self.params = dict()
        self.num_nodes = None
        self.instance_index = 0

    def generate(self):
        index = self.instance_index
        self.instance_index += 1
        name = type(self).__name__ + "_" + str(index)
        graph = self.generator()
        return name, graph


class CycleGraph(GraphGenerator):
    def __init__(self, N):
        generator = lambda seed: nx.cycle_graph(N)
        super(CycleGraph, self).__init__(generator)
        self.num_nodes = N
        self.params["N"] = N


class CompleteGraph(GraphGenerator):
    def __init__(self, N):
        generator = lambda seed: nx.complete_graph(N)
        super(CompleteGraph, self).__init__(generator)
        self.num_nodes = N
        self.params["N"] = N


class StarGraph(GraphGenerator):
    def __init__(self, N):
        generator = lambda seed: nx.star_graph(N - 1)
        super(StarGraph, self).__init__(generator)
        self.num_nodes = N
        self.params["N"] = N


class EmptyGraph(GraphGenerator):
    def __init__(self, N):
        generator = lambda: nx.empty_graph(N)
        super(EmptyGraph, self).__init__(generator)
        self.num_nodes = N
        self.params["N"] = N


class RegularGraph(GraphGenerator):
    def __init__(self, N, degree):
        generator = lambda: nx.random_regular_graph(
            N, degree, seed=np.random.randint(2 ** 31)
        )

        super(RegularGraph, self).__init__(generator)
        self.num_nodes = N
        self.params["N"] = N
        self.params["degree"] = degree


class BAGraph(GraphGenerator):
    def __init__(self, N, M):
        generator = lambda: nx.barabasi_albert_graph(
            N, M, seed=np.random.randint(2 ** 31)
        )
        super(BAGraph, self).__init__(generator)
        self.num_nodes = N
        self.params["N"] = N
        self.params["M"] = M


class ERGraph(GraphGenerator):
    def __init__(self, N, p):
        generator = lambda: nx.gnp_random_graph(N, p, seed=np.random.randint(2 ** 31))
        super(ERGraph, self).__init__(generator)
        self.num_nodes = N
        self.params["N"] = N
        self.params["p"] = p


class PartyMix(GraphGenerator):
    def __init__(self, generators):
        self.generators = generators
        generator = lambda: self._sample_generator()
        super(PartyMix, self).__init__(generator)
        self.params = dict()
        for g in self.generators:
            self.params.update(g.params)

    def _sample_generator(self):
        gen = choice(self.generators)
        n, g = gen.generate()
        return g
