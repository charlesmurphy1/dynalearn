import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    LeakyReLU,
    Dense,
    BatchNormalization,
    Activation,
)
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from .base import GNNModel
from .layers import GraphAttention


class Predictor(GNNModel):
    def __init__(self, config):

        super(Predictor, self).__init__()

        self.num_states = config.num_states
        self.in_features = config.in_features
        self.attn_features = config.attn_features
        self.out_features = config.out_features
        self.n_heads = config.n_heads
        self.in_activation = config.in_activation
        self.attn_activation = config.attn_activation
        self.out_activation = config.out_activation
        self.weight_decay = config.weight_decay
        self.tf_seed = config.tf_seed
        self.__config = config

    def _prepare_model(self):
        inputs = Input(shape=(1,))
        adj = Input(shape=(self.num_nodes,))

        x = Dense(
            self.in_features[0],
            activation="linear",
            kernel_initializer=glorot_uniform(self.tf_seed),
        )(inputs)
        x = Activation(self.in_activation)(x)
        for i in range(1, len(self.in_features)):
            x = Dense(
                self.in_features[i],
                activation=self.in_activation,
                kernel_initializer=glorot_uniform(self.tf_seed),
            )(x)
        for i in range(len(self.attn_features)):
            attn = GraphAttention(
                self.attn_features[i],
                attn_heads=self.n_heads[i],
                attn_heads_reduction="concat",
                kernel_initializer=glorot_uniform(self.tf_seed),
                attn_kernel_initializer=glorot_uniform(self.tf_seed),
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
                kernel_initializer=glorot_uniform(self.tf_seed),
            )(x)

        outputs = Dense(
            self.num_states,
            activation="softmax",
            kernel_initializer=glorot_uniform(self.tf_seed),
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


class EpidemicPredictor(Predictor):
    def __init__(self, config):
        super(EpidemicPredictor, self).__init__(config)

    @staticmethod
    def loss_fct(y_true, y_pred):
        return categorical_crossentropy(y_true, y_pred)
