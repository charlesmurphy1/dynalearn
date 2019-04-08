import tensorflow as tf


class MarkovPredictor():
    
    def __init__(self):
        self._model = None
        self.params = {}
        return    
    
    @property
    def model(self):
        if self._model != None:
            return self._model
        else:
            self._model = self.get_model()
            return self._model

    def get_model(self):
        raise NotImplemented("get_model() must be implemented")
    

    def predict(self, inputs, **kwargs):
        if self._model != None:
            return self._model.predict(inputs, **kwargs)
        else:
            self._model = self.get_model()
            return self._model.predict(inputs, **kwargs)

    def summary(self):
        if self._model != None:
            return self._model.summary()
        else:
            self._model = self.get_model()
            return self._model.summary()
