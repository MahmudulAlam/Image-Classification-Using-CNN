import pickle
import random
import tensorflow as tf


def to_one_hot(x, classes):
    return tf.one_hot(x[:, 0], classes)


def message(text):
    print("\033[92m" + text + "\033[0m")


def save_weights(file_name, weights):
    with open(file_name, 'wb') as f:
        pickle.dump(weights, f)


def load_weights(file_name):
    with open(file_name, 'rb') as f:
        weights = pickle.loads(f.read())
    return weights


def gen_indices(batch_size, dataset_size):
    index_a = list(range(0, dataset_size, batch_size))
    index_b = list(range(batch_size, dataset_size, batch_size))
    index_b.append(dataset_size)
    indices = list(zip(index_a, index_b))
    return indices


def shuffle(x, y):
    seed = random.randint(0, 1000)
    tf.random.set_seed(seed)
    x = tf.random.shuffle(x)
    tf.random.set_seed(seed)
    y = tf.random.shuffle(y)
    return x, y


def convert_matrix_to_tensor(mat, expand_dims='both'):
    # matrix of any 2d shape: mat : shape (m x n)
    # axis in which to expand dimension: expand_dims : string ('both', 'left', 'right)
    mat = tf.cast(tf.convert_to_tensor(mat), tf.float32)

    if expand_dims is 'both':
        mat = tf.expand_dims(mat, axis=0)
        mat = tf.expand_dims(mat, axis=-1)

    elif expand_dims is 'left':
        mat = tf.expand_dims(mat, axis=0)
        mat = tf.expand_dims(mat, axis=0)

    elif expand_dims is 'right':
        mat = tf.expand_dims(mat, axis=-1)
        mat = tf.expand_dims(mat, axis=-1)

    else:
        raise ValueError(
            expand_dims + "as expand_dims is invalid. Direction of dimension expansion is not valid. Try 'both' to "
                          "expand in both direction, 'left' to expand in only left direction, and 'right' to expand "
                          "in only right direction.")

    return mat


class NeatPrinter:
    def __init__(self, epoch):
        self.epoch = epoch
        self.epoch_len = len(str(epoch)) + 1
        self.progress_bar_len = 35
        self.sample = 'Epoch: {0:<' + str(self.epoch_len) + \
                      'd} [{1:<' + str(self.progress_bar_len) + \
                      's}] {2:>3d}%  >>  loss: {3:<.8f}  acc: {4:<.2f}'

    def print(self, iteration, loss, acc):
        percentage = int((iteration / self.epoch) * 100)
        scale = int((self.progress_bar_len / 100) * percentage)
        print(self.sample.format(iteration, '=' * scale + '.' * (self.progress_bar_len - scale), percentage, loss, acc))
