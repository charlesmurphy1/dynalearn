import numpy as np
from tensorflow.keras.layers import Input, Dense, Softmax, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from dynalearn.models.base import MarkovPredictor
from dynalearn.models.layers import GraphAttention


class GATBinaryMarkovPredictor(MarkovPredictor):
    def __init__(self, graph, n_hidden, n_heads,
                 weight_decay=1e-4, dropout=0.6, **kwargs):
        super(GATBinaryMarkovPredictor, self).__init__(**kwargs)
        
        if type(n_hidden) is int:
            n_hidden = [n_hidden]
        if type(n_heads) is int:
            n_heads = [n_heads] * len(n_hidden)
        elif len(n_heads) != len(n_hidden):
            raise ValueError
        
        self.graph = graph
        self.n_nodes = graph.number_of_nodes()
        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.weight_decay = weight_decay
        self.dropout = dropout

        self.params["name"] = type(self).__name__
        self.params["n_hidden"] = self.n_hidden
        self.params["n_heads"] = self.n_heads
        self.params["weight_decay"] = self.weight_decay
        self.params["dropout"] = self.dropout



    def get_model(self):
        N = self.graph.number_of_nodes()
        inputs = Input(shape=(1, ))
        adj = Input(shape=(N, ))

        x = inputs
        for i in range(len(self.n_hidden)):
            att = GraphAttention(self.n_hidden[i],
                                 attn_heads=self.n_heads[i],
                                 attn_heads_reduction='concat',
                                 dropout_rate=self.dropout,
                                 activation='linear',
                                 kernel_regularizer=l2(self.weight_decay))
            x = att([x, adj])

        outputs = Dense(1, activation='sigmoid')(x)

        return Model([inputs, adj], outputs=outputs)