import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    LeakyReLU,
    Dense,
    BatchNormalization,
    Activation,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import glorot_uniform

from .base import DynamicsPredictor
from .layers import GraphAttention


class LocalStatePredictor(DynamicsPredictor):
    def __init__(self, num_nodes, num_states, params=None, seed=None, **kwargs):

        super(LocalStatePredictor, self).__init__(
            num_nodes, num_states, params, **kwargs
        )

        self.in_features = params["in_features"]
        self.attn_features = params["attn_features"]
        self.out_features = params["out_features"]
        self.n_heads = params["n_heads"]
        self.in_activation = params["in_activation"]
        self.attn_activation = params["attn_activation"]
        self.out_activation = params["out_activation"]
        self.weight_decay = params["weight_decay"]
        self.seed = seed

    def _prepare_model(self):
        inputs = Input(shape=(1,))
        adj = Input(shape=(self.num_nodes,))

        x = Dense(
            self.in_features[0],
            activation="linear",
            kernel_initializer=glorot_uniform(self.seed),
        )(inputs)
        x = Activation(self.in_activation)(x)
        for i in range(1, len(self.in_features)):
            x = Dense(
                self.in_features[i],
                activation=self.in_activation,
                kernel_initializer=glorot_uniform(self.seed),
            )(x)
        for i in range(len(self.attn_features)):
            attn = GraphAttention(
                self.attn_features[i],
                attn_heads=self.n_heads[i],
                attn_heads_reduction="concat",
                kernel_initializer=glorot_uniform(self.seed),
                attn_kernel_initializer=glorot_uniform(self.seed),
                dropout_rate=0,
                activation="linear",
                kernel_regularizer=l2(self.weight_decay),
            )
            x, attn_coeff = attn([x, adj])
            x = Activation(self.attn_activation)(x)

        for i in range(len(self.out_features)):
            x = Dense(
                self.out_features[i],
                activation=self.out_activation,
                kernel_initializer=glorot_uniform(self.seed),
            )(x)

        outputs = Dense(
            self.num_states,
            activation="softmax",
            kernel_initializer=glorot_uniform(self.seed),
        )(x)

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
