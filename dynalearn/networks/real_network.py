import networkx as nx
import numpy as np

from dynalearn.networks.base import Network
from dynalearn.config import Config


class RealNetwork(Network):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        Network.__init__(self, config)
        self.data = [nx.from_edgelist(config.edgelist)]
        self._num_nodes = self.data[0].number_of_nodes()

    def generate(self):
        return self.data[0]


class RealTemporalNetwork(Network):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = Config()
            config.__dict__ = kwargs
        Network.__init__(self, config)
        self.edges = config.edges
        self.window = config.window
        self.dt = config.dt
        self.all_edges = [(int(e[1]), int(e[2])) for e in self.edges]
        complete_network = nx.from_edgelist(self.all_edges)
        self.all_nodes = list(complete_network.nodes())
        self.node_map = {n: i for i, n in enumerate(self.all_nodes)}
        self._num_nodes = len(self.all_nodes)
        self.edgelist, self.times = RealTemporalNetwork.format_edgelist(
            self.edges, self.dt, lambda i: self.node_map[i]
        )
        self.complete_network = nx.relabel_nodes(complete_network, self.node_map)
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
                    self.edgelist,
                    self.all_nodes,
                    i * self.window,
                    self.window,
                    self.num_nodes,
                )
            )
        return network_list

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, time):
        self._time = time
        if self._time == len(self.data):
            self._time = 0

    @staticmethod
    def format_edgelist(edges, dt, mapping=lambda i: i):
        first = edges[0, 0]
        last = edges[-1, 0]
        num_timesteps = int((last - first) / dt)
        edge_lists = [[] for i in range(num_timesteps + 1)]
        times = np.arange(first, last + dt, dt)

        index = 0
        for e in edges:
            while times[index] != e[0]:
                index += 1
            edge_lists[index].append((mapping(e[1]), mapping(e[2])))
        return edge_lists, times

    @staticmethod
    def make_network(edgelist, nodes, t, window, num_nodes):
        edges = []
        g = nx.empty_graph(num_nodes)
        for i in range(window):
            for e in edgelist[t + i]:
                g.add_edge(e[0], e[1])
            if t + i == len(edgelist) - 1:
                break
        return g
