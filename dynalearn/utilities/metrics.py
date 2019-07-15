import tensorflow.keras.backend as K


def model_entropy(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return -K.sum(y_pred * K.log(y_pred), axis=-1)


def approx_kl_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return -K.sum(y_true * K.log(y_pred), axis=-1) + K.sum(
        y_pred * K.log(y_pred), axis=-1
    )
