import numpy as np
from tensorflow.keras.layers import Input, LeakyReLU, Dense, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import glorot_uniform

from .base import DynamicsPredictor
from .layers import GraphAttention


class LocalStatePredictor(DynamicsPredictor):
    def __init__(self, num_nodes, num_states, n_hidden, n_heads,
                 weight_decay=1e-4, dropout=0.6, bn_momentum=0.99,
                 bn_epsilon=0.0001, seed=None,
                 **kwargs):
        
        super(LocalStatePredictor, self).__init__(num_nodes,
                                                       num_states,
                                                       **kwargs)
        
        if type(n_hidden) is int:
            n_hidden = [n_hidden]
        if type(n_heads) is int:
            n_heads = [n_heads] * len(n_hidden)
        elif len(n_heads) != len(n_hidden):
            raise ValueError
        
        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon
        self.seed = seed

        self.params["n_hidden"] = self.n_hidden
        self.params["n_heads"] = self.n_heads
        self.params["weight_decay"] = self.weight_decay
        self.params["dropout"] = self.dropout
        self.params["bn_momentum"] = self.bn_momentum
        self.params["bn_epsilon"] = self.bn_epsilon



    def _prepare_model(self):
        inputs = Input(shape=(1, ))
        adj = Input(shape=(self.num_nodes, ))

        x = Dense(self.n_hidden[0], activation='tanh',
                  kernel_initializer=glorot_uniform(self.seed))(inputs)

        for i in range(len(self.n_hidden)):
            attn = GraphAttention(self.n_hidden[i], 8,
                                  attn_heads=self.n_heads[i],
                                  attn_heads_reduction='concat',
                                  kernel_initializer=glorot_uniform(self.seed),
                                  attn_kernel_initializer=glorot_uniform(self.seed),
                                  dropout_rate=self.dropout,
                                  activation='linear',
                                  kernel_regularizer=l2(self.weight_decay))
            x, attn_coeff = attn([x, adj])

            x = BatchNormalization(momentum=self.bn_momentum,
                                   epsilon=self.bn_epsilon)(x)
            x = Activation('relu')(x)

        outputs = Dense(self.num_states, activation='softmax', 
                        kernel_initializer=glorot_uniform(self.seed))(x)

        return Model([inputs, adj], outputs=outputs)


    def get_attn_layer(self, depth):
        inputs = self.model.input
        i = 0
        for layer in self.model.layers:
            if type(layer) == GraphAttention:
                i += 1

            if i == depth:
                features, attn_coeff = layer.output
                break

        return Model(inputs, attn_coeff)




