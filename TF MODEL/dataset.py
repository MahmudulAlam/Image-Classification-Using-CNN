import tensorflow as tf
from keras.datasets import cifar10


def load_dataset():
    classes = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = tf.convert_to_tensor(x_train / 255., tf.float32)
    x_test = tf.convert_to_tensor(x_test / 255., tf.float32)

    y_train = tf.convert_to_tensor(tf.keras.utils.to_categorical(y_train, classes), tf.float32)
    y_test = tf.convert_to_tensor(tf.keras.utils.to_categorical(y_test, classes), tf.float32)

    # y_train = tf.transpose(y_train)
    # y_test = tf.transpose(y_test)

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = load_dataset()
    print('train_x:', train_x.shape)
    print('train_y:', train_y.shape)
    print('test_x:', test_x.shape)
    print('test_y:', test_y.shape)
