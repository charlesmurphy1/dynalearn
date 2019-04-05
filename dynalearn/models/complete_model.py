import numpy as np
from tensorflow.keras.layers import Input, Dense, Softmax, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from dynalearn.models.base import MarkovPredictor


class CompleteMarkovPredictor(MarkovPredictor):
    def __init__(self, graph, n_hidden, weight_decay=1e-4, **kwargs):
        super(CompleteMarkovPredictor, self).__init__(**kwargs)
        self.graph = graph
        self.n_hidden = n_hidden
        self.weight_decay = weight_decay


    def get_model(self):
        N = self.graph.number_of_nodes()
        inputs = Input(shape=(N, ))
        h = Dense(self.n_hidden[0],
                  activation='relu',
                  kernel_regularizer=l2(self.weight_decay))(inputs)
        for i in range(1, len(self.n_hidden)):
            h = Dense(self.n_hidden[i],
                      activation='relu',
                      kernel_regularizer=l2(self.weight_decay))(h)

        outputs = Dense(N, activation='sigmoid')(h)

        return Model(inputs, outputs)