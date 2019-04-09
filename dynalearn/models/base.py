# import tensorflow as tf


# class MarkovPredictor():
    
#     def __init__(self):
#         self._model = None
#         return    
    
#     @property
#     def model(self):
#         if self._model != None:
#             return self._model
#         else:
#             self._model = self.get_model()
#             return self._model

#     def get_model(self):
#         raise NotImplemented("get_model() must be implemented")




import tensorflow as tf


class MarkovPredictor():
    
    def __init__(self, num_nodes):
        self._model = None
        self._num_nodes = num_nodes
        self.params = {}
    
    @property
    def model(self):
        if self._model is None:
            self._model = self._prepare_model()
        return self._model

    def _prepare_model(self):
        raise NotImplemented("_prepare_model() must be implemented")


    @property
    def num_nodes(self):
        if self._num_nodes is None:
            raise ValueError('No graph has been given to the dynamics.')
        else:
            return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes):
        if self._num_nodes != num_nodes:
            self._num_nodes = num_nodes
            if self._model is None:
                self._model = self._prepare_model()
            else:
                weights = self._model.get_weights()
                self._model = self._prepare_model()
                self._model.set_weights(weights)
    