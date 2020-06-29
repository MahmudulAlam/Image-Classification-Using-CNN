import tensorflow as tf


def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))


def accuracy_fn(y_true, y_pred):
    y_true = tf.cast(tf.argmax(y_true, axis=-1), tf.float32)
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
    compare = tf.cast(tf.equal(y_true, y_pred), tf.float32)
    accuracy = tf.reduce_mean(compare) * 100
    return accuracy


def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.math.squared_difference(y_true, y_pred))


def binary_cross_entropy_loss(y_true, y_pred):
    return tf.reduce_mean(- y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred))


def categorical_cross_entropy_loss(y_true, y_pred):
    return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))
