from keras.models import Input, Model
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

alpha = 0.1


def conv_with_batch_norm(x, kernel, size, padding, activation):
    x = Conv2D(kernel, size, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == 'leaky':
        x = LeakyReLU(alpha)(x)
    else:
        x = Activation('relu')(x)
    return x


def model():
    input = Input(shape=(32, 32, 3))

    # Block 01
    x = conv_with_batch_norm(input, 32, (3, 3), padding='same', activation='leaky')
    x = conv_with_batch_norm(x, 32, (1, 1), padding='same', activation='leaky')
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Block 02
    x = conv_with_batch_norm(x, 64, (3, 3), padding='same', activation='leaky')
    x = conv_with_batch_norm(x, 64, (1, 1), padding='same', activation='leaky')
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Block 03
    x = conv_with_batch_norm(x, 64, (3, 3), padding='same', activation='leaky')
    x = conv_with_batch_norm(x, 64, (1, 1), padding='same', activation='leaky')
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Block 04
    x = conv_with_batch_norm(x, 128, (3, 3), padding='same', activation='leaky')
    x = conv_with_batch_norm(x, 128, (1, 1), padding='same', activation='leaky')
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Block 05
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)

    return Model(inputs=input, outputs=x)


if __name__ == '__main__':
    model = model()
    model.summary()
