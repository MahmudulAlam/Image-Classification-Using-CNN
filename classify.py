import os
import numpy as np
from PIL import Image
from model import model
import matplotlib.pyplot as plt

classes_name = {0: 'Airplane', 1: 'Automobile', 2: 'Bird', 3: 'Cat', 4: 'Deer', 5: 'Dog', 6: 'Frog', 7: 'Horse',
                8: 'Ship', 9: 'Truck'}

model = model()
model.summary()
model.load_weights('MODEL DATA/model.h5')


def classify(image):
    img = image.resize((32, 32))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    prediction = model.predict(img)
    prediction = prediction[0]
    max_index = np.argmax(prediction)
    class_name = classes_name.get(max_index)
    return class_name


image_names = os.listdir('IMAGES/')
for n, image_name in enumerate(image_names, 1):
    image = Image.open('IMAGES/' + image_name)
    class_name = classify(image=image)
    plt.subplot(5, 4, n)
    plt.imshow(image)
    plt.title(class_name, fontsize=10)

plt.show()
