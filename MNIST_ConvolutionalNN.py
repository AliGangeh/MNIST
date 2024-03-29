#import libraries
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dropout
from keras.models import Model
import random
import cv2
import requests
from PIL import Image

#loads mnist data
np.random.seed(0)
(X_train, y_train), (X_test, y_test)= mnist.load_data()

#checks for errors
assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels."
assert(X_train.shape[1:] == (28,28)), "The dimensions of the images are not 28 x 28."
assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not equal to the number of labels."
assert(X_test.shape[1:] == (28,28)), "The dimensions of the images are not 28 x 28."

#displays a sample set of the MNIST
num_of_samples=[]
cols = 5
num_classes = 10
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5,10))
fig.tight_layout()
for i in range(cols):
    for j in range(num_classes):
      x_selected = X_train[y_train == j]
      axs[j][i].imshow(x_selected[random.randint(0,(len(x_selected) - 1)), :, :], cmap=plt.get_cmap('gray'))
      axs[j][i].axis("off")
      if i == 2:
        axs[j][i].set_title(str(j))
        num_of_samples.append(len(x_selected))

#displays distribution
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the train dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

#formats MNIST input and labels so its usabel by the Neural Network
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
X_train = X_train/255
X_test = X_test/255

#definition of creates a convolutional neural network with dropout (which reduces loss and stops overfitting)
def leNet_model():
    model = Sequential()
    model.add(Conv2D(30, (5,5), input_shape=(28, 28, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(Adam(lr=0.01), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

#runs model
model = leNet_model()
print(model.summary())
history = model.fit(X_train, y_train, epochs=10, validation_split=0.1, batch_size=400, verbose=1, shuffle=1)

#plots loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["loss", "val_loss"])
plt.title("loss")
plt.xlabel("epochs")
plt.show()

#plots accuracy
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.legend(["acc", "val_acc"])
plt.title("accuracy")
plt.xlabel("epochs")
plt.show()

#tests generalization to see how good it is with data it wasn't trained with it has 92.88% accuracy
score= model.evaluate(X_test, y_test, verbose=0)
print(type(score))
print("test score:", score[0])
print("test accuracy", score[1])

#imports image and makes it usable for the model
url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcST8KzXHtkSHcxzdpnllMhAj0upLEwnNFdtY6j4YUPcmaf4Ty3u'
response = requests.get(url, stream=True)
img = Image.open(response.raw)
plt.imshow(img, cmap=plt.get_cmap("gray"))

#processes image so it can be inputted in the model
img = np.asarray(img)
img = cv2.resize(img, (28, 28))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.bitwise_not(img)
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()
img = img/255
img = img.reshape(1, 28, 28, 1)

#model predicts what the new input is
prediction = model.predict_classes(img)
print("predicted digit:", str(prediction))

#model accuracy and loss is printed
score= model.evaluate(X_test, y_test, verbose=0)
print(type(score))
print("test score:", score[0])
print("test accuracy", score[1])

#displays what the images in layer 1 and layer 2 look like
layer1 = Model(inputs=model.layers[0].input, outputs=model.layers[0].output)
layer2 = Model(inputs=model.layers[0].input, outputs=model.layers[2].output)
visual_layer1, visual_layer2 = layer1.predict(img), layer2.predict(img)
#layer 1
plt.figure(figsize=(10, 6))
for i in range(30):
    plt.subplot(6, 5, i+1)
    plt.imshow(visual_layer1[0, :, :, i], cmap=plt.get_cmap('jet'))
    plt.axis('off')
plt.show()
#layer 2
plt.figure(figsize=(10, 6))
for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.imshow(visual_layer2[0, :, :, i], cmap=plt.get_cmap('jet'))
    plt.axis('off')
plt.show()
