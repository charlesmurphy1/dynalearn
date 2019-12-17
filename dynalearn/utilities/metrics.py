import tensorflow.keras.backend as K


def model_entropy(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return -K.sum(y_pred * K.log(y_pred), axis=-1)


def relative_entropy(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.categorical_crossentropy(y_true, y_pred) - K.categorical_crossentropy(
        y_true, y_true
    )


def approx_relative_entropy(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.categorical_crossentropy(y_true, y_pred) - model_entropy(y_true, y_pred)


def jensenshannon(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    m = 0.5 * (y_true + y_pred)
    return 0.5 * (relative_entropy(y_true, m) + relative_entropy(y_pred, m))


all_metrics = {
    "model_entropy": model_entropy,
    "relative_entropy": relative_entropy,
    "approx_relative_entropy": approx_relative_entropy,
    "jensenshannon": jensenshannon,
}


def get_metrics(name):
    if name in all_metrics:
        return all_metrics[name]
    else:
        raise ValueError(
            "Wrong name of metrics. Valid entries are: {0}".format(
                list(all_metrics.keys())
            )
        )
