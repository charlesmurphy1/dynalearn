import networkx as nx
import numpy as np
import h5py


class GraphGenerator:
    def __init__(self, generator):
        self.generator = generator
        self.params = dict()
        self.instance_index = 0


    def generate(self):
        index = self.instance_index
        self.instance_index += 1
        name = type(self).__name__ + '_' + str(index)
        graph = self.generator()
        return name, graph


class CycleGraph(GraphGenerator):
    def __init__(self, N, rand_gen=None):
        generator = lambda: nx.cycle_graph(N)
        super(CycleGraph, self).__init__(generator)
        self.params['N'] = N


class CompleteGraph(GraphGenerator):
    def __init__(self, N, rand_gen=None):
        generator = lambda: nx.complete_graph(N)
        super(CompleteGraph, self).__init__(generator)
        self.params['N'] = N


class StarGraph(GraphGenerator):
    def __init__(self, N, rand_gen=None):
        generator = lambda: nx.star_graph(N)
        super(StarGraph, self).__init__(generator)
        self.params['N'] = N


class EmptyGraph(GraphGenerator):
    def __init__(self, N, rand_gen=None):
        generator = lambda: nx.empty_graph(N)
        super(EmptyGraph, self).__init__(generator)
        self.params['N'] = N


class RegularGraph(GraphGenerator):
    def __init__(self, N, degree, rand_gen=None):
        generator = lambda: nx.random_regular_graph(N, degree, seed=rand_gen)

        super(RegularGraph, self).__init__(generator)
        self.params['N'] = N
        self.params['degree'] = degree


class BAGraph(GraphGenerator):
    def __init__(self, N, M, rand_gen=None):
        generator = lambda: nx.barabasi_albert_graph(N, M, seed=rand_gen)
        super(BAGraph, self).__init__(generator)
        self.params['N'] = N
        self.params['M'] = M


class ERGraph(GraphGenerator):
    def __init__(self, N, p, rand_gen=None):
        generator = lambda: nx.gnp_random_graph(N, p, seed=rand_gen)
        super(ERGraph, self).__init__(generator)
        self.params['N'] = N
        self.params['p'] = p
