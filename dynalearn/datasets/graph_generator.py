import networkx as nx
import h5py


class GraphGenerator(nx.Graph):
    def __int__(self, name, generator):
        self.name = name
        self.generator = generator
        self.graphs = []


    def generate(self, num_sample):
        for i in range(num_sample):
            self.graphs.append(self.generator())

    def save_graphs(self, path=None):
        if path is None:
            path = './' + self.name

        h5file = h5py.File(path, "w")
        for i in range(len(self.graphs)):
            g = self.graphs[i]
            edgelist = np.array(g.edges)
            h5file.create_dataset("{0}{1}/edgelist".format(self.name, i),
                                  data=edgelist, dtype='i')

        h5file.close()

    def load_graphs(self, path):
        h5file = h5py.File(path, "r")
        for k in h5file.keys():
            edgelist = h5file[k + '/edgelist'][...]
            g = nx.from_edgelist(edgelist)
            self.graphs.append(g)

        h5file.close()


class CycleGraph(GraphGenerator):
    def __init__(self, N):
        self.params['N'] = N
        name = 'cycle_n{0}'.format(N)
        generator = lambda: nx.cycle_graph(N)
        super(CycleGraph, self).__init__(name, generator)


class CompleteGraph(GraphGenerator):
    def __init__(self, N):
        self.params['N'] = N
        name = 'complete_n{0}'.format(N)
        generator = lambda: nx.complet_graph(N)
        super(CompleteGraph, self).__init__(name, generator)


class StarGraph(GraphGenerator):
    def __init__(self, N):
        self.params['N'] = N
        name = 'star_n{0}'.format(N)
        generator = lambda: nx.star_graph(N)
        super(StarGraph, self).__init__(name, generator)


class RegularGraph(GraphGenerator):
    def __init__(self, N, degree):
        self.params['N'] = N
        self.params['degree'] = degree
        name = 'regular_n{0}_d{1}'.format(N, degree)
        generator = lambda: nx.random_regular_graph(N, degree)
        super(RegularGraph, self).__init__(name, generator)


class EmptyGraph(GraphGenerator):
    def __init__(self, N):
        self.params['N'] = N
        name = 'empty_n{0}'.format(N)
        generator = lambda: nx.empty_graph(N)
        super(EmptyGraph, self).__init__(name, generator)


class BAGraph(GraphGenerator):
    def __init__(self, N, M):
        self.params['N'] = N
        self.params['M'] = M
        name = 'BA_n{0}_m{1}'.format(N, M)
        generator = lambda: nx.barabasi_albert_graph(N, M)
        super(BA_Graph, self).__init__(name, generator)


class ERGraph(GraphGenerator):
    def __init__(self, N, p):
        self.params['N'] = N
        self.params['p'] = p
        name = 'ER_n{0}_p{1}'.format(N, p)
        generator = lambda: nx.gnp_random_graph(N, p)
        super(ER_Graph, self).__init__(name, generator)
