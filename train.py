from keras.datasets import cifar10
from keras.utils import to_categorical
from model import model

classes = 10

""" Loading the dataset """
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

""" Normalization """
x_train = x_train / 255.
x_test = x_test / 255.

""" From numerical label to categorical label """
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

""" Building the Model """
model = model()
model.summary()
""" End of the Model """

""" Compilation """
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

""" Training """
history = model.fit(x_train, y_train, batch_size=256, epochs=120, verbose=2, shuffle=True,
                    validation_data=(x_test, y_test))

""" Evaluation of the model """
score = model.evaluate(x_test, y_test, verbose=1)
print('Test set loss: {0:.2f} and Accuracy: {1:.2f}%'.format(score[0], score[1] * 100))

""" Saving the weight file """
model.save_weights('MODEL DATA/cifar-10.h5')

""" Saving the history in a text file """
with open('history.txt', 'a+') as f:
    print(history.history, file=f)
    print(score, file=f)

print('All Done!')
