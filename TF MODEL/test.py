from model import model
from dataset import load_dataset
from cnn.utils import gen_indices
from cnn.metrics import accuracy_fn

x_train, y_train, x_test, y_test = load_dataset()

model = model()
model.load_weights('../MODEL DATA/tf-model.h5')

indices = gen_indices(512, x_test.shape[0])
accuracy = []
for (a, b) in indices:
    y_pred = model(x_test[a:b])
    y_true = y_test[a:b]
    acc = accuracy_fn(y_true, y_pred)
    accuracy.append(acc.numpy())

accuracy = sum(accuracy) / len(accuracy)
print('Accuracy: {0:.2f}%'.format(accuracy))
