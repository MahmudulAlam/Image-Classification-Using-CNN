from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.layers import LeakyReLU

alpha = 0.1


def model():
    """ Building Model """
    model = Sequential()

    # Block 01
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Conv2D(32, (3, 3)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 02
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Conv2D(64, (3, 3)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 03
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Conv2D(64, (3, 3)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Block 04
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model


if __name__ == '__main__':
    model = model()
    model.summary()
