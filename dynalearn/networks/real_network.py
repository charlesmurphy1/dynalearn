import networkx as nx

from dynalearn.networks.base import Network
from dynalearn.config import Config


class RealNetwork(Network):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        Network.__init__(config)
        self.data = [nx.from_edgelist(config.edgelist)]

    def generate(self):
        return self.data[0]


class RealTemporalNetwork(Network):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        Network.__init__(config)
        self.edges = config.edges
        self.window = config.window
        self.dt = config.dt
        self.all_edges = [(int(e[0]), int(e[1])) for e in self.edges]
        self.complete_network = nx.from_edgelist(self.all_edges)
        self.all_nodes = list(self.network_all.nodes())
        self.edgelist, self.times = RealTemporalGraph.format_edgelist(
            self.edges, self.dt
        )
        self.data = self.get_network_list()
        self.time = 0

    def generate(self):
        g = self.data[self.time]
        self.time += 1
        return g

    def get_network_list(self):
        num_networks = int(len(self.edgelist) / self.window)
        network_list = []
        for i in range(num_networks):
            network_list.append(
                RealTemporalNetwork.make_network(
                    self.edgelist, self.all_nodes, i * self.window, self.window
                )
            )
        return network_list

    @staticmethod
    def format_edgelist(edges, dt):
        first = edges[0, -1]
        last = edges[-1, -1]
        num_timesteps = int((last - first) / dt)
        edge_lists = [[] for i in range(num_timesteps + 1)]
        times = np.arange(first, last + dt, dt)

        index = 0
        for e in edges:
            while times[index] != e[-1]:
                index += 1
            edge_lists[index].append((int(e[0]), int(e[1])))
        return edge_lists, times

    @staticmethod
    def make_network(edgelist, nodes, t, window):
        edges = []
        g = nx.Graph()
        g.add_nodes_from(nodes)
        for i in range(window):
            for e in edgelist[t + i]:
                g.add_edge(e[0], e[1])
            if t + i == len(edgelist) - 1:
                break
        return g
