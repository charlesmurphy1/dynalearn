import torch
from torch.utils.data.dataset import Dataset
import networkx as nx
from random import shuffle


class Node_State_Dataset(Dataset):
    """docstring for Node_State_Dataset"""
    def __init__(self, states, memory=1, conversion_function=lambda x: x):
        super(Node_State_Dataset, self).__init__()
        self.states = conversion_function(states)
        self.T = len(states)
        self.memory = memory


    def __getitem__(self, index):

        present_state = self.states[index]
        passed_states = [None] * self.memory
        time = [None] * self.memory

        for i in range(1, self.memory + 1):
            passed_states[i - 1] = self.states[index - i]

        return present_state, passed_states

    def __len__(self):
        return self.T - memory


class Dynamical_Network_Dataset(Dataset):
    def __init__(self,
                 path_to_states,
                 path_to_edgelist=None,
                 memory=1,
                 shuffle_neighbors=False,
                 conversion_function=lambda x:x):

        self.memory = memory
        self.shuffle_neighbors = shuffle_neighbors
        self.conversion_function = conversion_function

        complete_data = self.load_complete_data(path_to_states)
        self.load_network(path_to_edgelist)
        self.load_states(complete_data)

        if path_to_edgelist:
            self.item_function = self._getitem_with_structure
        else:
            self.item_function = self._getitem_without_structure

        return None


    def __getitem__(self, index):
        return self.item_function(index)


    def __len__(self):
        return self.num_node * (self.num_time - self.memory)


    def _getitem_with_structure(self, index):
        node, time = self.node_time_from_index(index)

        present_state, passed_state = self.state_data[node][time]

        neighbors_list = list(self.g.neighbors(node))
        if self.shuffle_neighbors:
            shuffle(neighbors_list)

        neighbors_passed_states = [self.state_data[n][time][1] for n in neighbors_list]

        return present_state, passed_state, neighbors_passed_states


    def _getitem_without_structure(self, index):

        present_states = [None] * self.num_node
        passed_states = [None] * self.num_node
        for n in self.g.nodes():
            states = self.state_data[n][index]
            present_states[n] = states[0]
            passed_states[n] = states[1]

        return present_states, passed_states


    def node_time_from_index(self, index):

        node_index = self.node_data[index % self.num_node]
        time_index = index // self.num_node + self.memory
        return node_index, time_index


    def load_complete_data(self, path_to_states):

        import pickle
        complete_data = []

        f = open(path_to_states, "rb")
        while True:
            try:
                complete_data.append(pickle.load(f))
            except:
                break
        f.close()

        self.num_node = len(complete_data[0][1])
        self.num_time = len(complete_data)

        return complete_data


    def load_states(self, complete_data):

        # Gathering the nodes' states
        self.state_data = {}
        for i in range(self.num_node):
            node =self.node_data[i]
            states = [None] * self.num_time
            for t in range(self.num_time):
                states[t] = complete_data[t][1][i]

            self.state_data[node] = Node_State_Dataset(states, self.memory,
                                                        self.conversion_function)


        # Gathering the times
        self.time_data = [None] * self.num_time
        for t in range(self.num_time):
            self.time_data[t] = complete_data[t][0]

        return None


    def load_network(self, path_to_edgelist=None):

        # Gathering graph
        if path_to_edgelist is not None:
            self.g = nx.read_edgelist(path_to_edgelist)
        else:
            self.g = nx.empty_graph(self.num_node)

        # Gathering nodes and indexes
        self.node_data = {}
        for i, node in enumerate(self.g.nodes()):
            self.node_data[i] = node

        return None