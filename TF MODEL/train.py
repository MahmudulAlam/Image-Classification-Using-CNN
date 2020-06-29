import keras
import random
import tensorflow as tf
from model import model
from dataset import load_dataset
from cnn.metrics import loss_fn, accuracy_fn
from cnn.utils import shuffle, gen_indices, NeatPrinter

model = model()
model.summary()
model.load_weights('../MODEL DATA/tf-model.h5')

x_train, y_train, x_test, y_test = load_dataset()
optimizer = keras.optimizers.Adam(lr=1e-6)

# params
epochs = 100
batch_size = 256
dataset_size = x_train.shape[0]
printer = NeatPrinter(epoch=epochs)
indices = gen_indices(batch_size, dataset_size)

for i, epoch in enumerate(range(epochs), 1):
    loss = []
    accuracy = []
    x_train, y_train = shuffle(x_train, y_train)

    for (a, b) in indices:
        x_true = x_train[a:b]
        y_true = y_train[a:b]

        with tf.GradientTape() as tape:
            y_pred = model(x_true, training=True)
            loss_value = loss_fn(y_true, y_pred)
            loss.append(loss_value.numpy())
            acc = accuracy_fn(y_true, y_pred)
            accuracy.append(acc.numpy())

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    loss_val = sum(loss) / len(loss)
    acc_val = sum(accuracy) / len(accuracy)
    printer.print(i, loss_val, acc_val)

model.save_weights('../MODEL DATA/tf-model.h5')
