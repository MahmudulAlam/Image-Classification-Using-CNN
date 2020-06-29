![image1](https://user-images.githubusercontent.com/37298971/38595828-bb62b8c4-3d6f-11e8-9179-4b6e92491838.png)
## <h1 align="center">Image Classification Using CNN</h1>
Image Classification Using CNN
Canadian Institute for Advanced Research (CIFAR) provides a dataset that consists of 60000 32x32x3 color images of 10 classes, known as CIFAR-10, with 6000 images per class. There are 50000 training images and 10000 test images. To classify those 10 classes of images a convolutional neural network (CNN) is used here. CNN achieved 85.0% accuracy in the test dataset. The block diagram of the CNN is shown below.
## Block Diagram
![block](https://user-images.githubusercontent.com/37298971/56470779-02ef6600-646c-11e9-8b86-a627ce471c94.png)
## CNN Architecture
```
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
```
Batch size is chosen 256 and the network is trained for 120 epochs. 

## Loss curve
![loss curve](https://user-images.githubusercontent.com/37298971/56470709-1f3ed300-646b-11e9-852b-5077532be4b3.png)


## Accuracy curve 
![acc curve](https://user-images.githubusercontent.com/37298971/56470707-1bab4c00-646b-11e9-8d97-b224ad66991d.png)



## Output
This is how the outputs look like.
![tesla](https://user-images.githubusercontent.com/37298971/45700294-e6b1e080-bb8d-11e8-9a19-0ce2b84c04ae.png)

![air](https://user-images.githubusercontent.com/37298971/45700302-ee718500-bb8d-11e8-9d44-46c8d8536a1a.png)

#### Output for all the images in the ```'IMAGES/'``` folder. 
![OUTPUT](https://user-images.githubusercontent.com/37298971/56470711-21a12d00-646b-11e9-9999-a85cceafbbf1.png)

## Summary
- [x] Load the CIFAR-10 dataset.
- [x] Normalize training and test data.
- [x] Change labels from integer to categorical.
- [x] Build the model.
- [x] Compile the model.
- [x] Train the model.
- [x] Save the model.
- [x] Classify new test image using the trained model.
