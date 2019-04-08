import tensorflow as tf


class MarkovPredictor():
    
    def __init__(self, graph):
        self._model = None
        self._graph = graph
        self._num_nodes = graph.number_of_nodes()
        self.params = {}
    
    @property
    def model(self):
        return self._model

    def prepare_model(self):
        raise NotImplemented("prepare_model() must be implemented")


    @property
    def graph(self):
        if self._graph is None:
            raise ValueError('No graph has been given to the dynamics.')
        else:
            return self._graph

    @graph.setter
    def graph(self, graph):
        self._graph = graph
        self._num_nodes = graph.number_of_nodes()
        if self._model is None:
            self._model = self.prepare_model()
        else:
            weights = self.model.get_weights()
            self._model = self.prepare_model()
            self.model.set_weights(weights)
    
