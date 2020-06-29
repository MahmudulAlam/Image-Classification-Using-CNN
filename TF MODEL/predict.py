import cv2
import numpy as np
import tensorflow as tf
from model import model
import matplotlib.pyplot as plt

classes_name = {0: 'Airplane', 1: 'Automobile', 2: 'Bird', 3: 'Cat', 4: 'Deer', 5: 'Dog', 6: 'Frog', 7: 'Horse',
                8: 'Ship', 9: 'Truck'}

model = model()
model.load_weights('../MODEL DATA/tf-model.h5')


def classify(img):
    img = cv2.resize(img, (32, 32)) / 255.0
    img = tf.expand_dims(tf.cast(img, tf.float32), axis=0)
    prediction = model.predict(img)[0]
    index = np.argmax(prediction)
    class_name = classes_name[int(index)]
    return class_name


image = plt.imread('../IMAGES/car1.jpg')
cls_name = classify(image)
plt.imshow(image)
plt.title(cls_name, fontsize=10)
plt.show()
