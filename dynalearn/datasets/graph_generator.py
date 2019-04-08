import networkx as nx
import h5py


class GraphGenerator(nx.Graph):
    def __int__(self, generator):
        self.generator = generator
        self.params = dict()
        self.instance_index = 0


    def generate(self, num_sample):
        index = self.instance_index
        self.instance_index += 1
        name = type(self).__name__ + '_' + str(index)
        graph = self.generator()
        return name, graph


class CycleGraph(GraphGenerator):
    def __init__(self, N):
        super(CycleGraph, self).__init__(lambda: nx.cycle_graph(N))
        self.params['N'] = N


class CompleteGraph(GraphGenerator):
    def __init__(self, N):
        super(CompleteGraph, self).__init__(lambda: nx.complet_graph(N))
        self.params['N'] = N


class StarGraph(GraphGenerator):
    def __init__(self, N):
        generator = lambda: nx.star_graph(N)
        super(StarGraph, self).__init__(lambda: nx.star_graph(N))
        self.params['N'] = N


class RegularGraph(GraphGenerator):
    def __init__(self, N, degree):
        generator = lambda: nx.random_regular_graph(N, degree)
        super(RegularGraph, self).__init__(lambda: nx.random_regular_graph(N, degree))
        self.params['N'] = N
        self.params['degree'] = degree


class EmptyGraph(GraphGenerator):
    def __init__(self, N):
        super(EmptyGraph, self).__init__(lambda: nx.empty_graph(N))
        self.params['N'] = N


class BAGraph(GraphGenerator):
    def __init__(self, N, M):
        super(BA_Graph, self).__init__(lambda: nx.barabasi_albert_graph(N, M))
        self.params['N'] = N
        self.params['M'] = M


class ERGraph(GraphGenerator):
    def __init__(self, N, p):
        super(ER_Graph, self).__init__(lambda: nx.gnp_random_graph(N, p))
        self.params['N'] = N
        self.params['p'] = p
