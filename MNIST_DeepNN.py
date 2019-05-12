#import libraries
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random

#saves mnist data x_train+ x_test is coordinates y_train and y_test are labels
np.random.seed(0)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#checks for error
assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels."
assert(X_test.shape[0] == y_test.shape[0]), "The number of images is not equal to the number of labels."
assert(X_train.shape[1:] == (28,28)), "The dimensions of the images are not 28x28"
assert(X_test.shape[1:] == (28,28)), "The dimensions of the images are not 28x28"

#prints out a sample of 5 drawings for each number
num_of_samples = []
cols = 5
num_classes = 10
fig, axs = plt.subplots(nrows=num_classes, ncols = cols, figsize=(5, 8))
fig.tight_layout()
for i in range(cols):
    for j in range(num_classes):
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected - 1)), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j))
            num_of_samples.append(len(x_selected))
plt.show()

#amount of drawing per number
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

#makes use of one-hot encoding because there are multiple outputs
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#because it is grayscale divided by 255 to get a normalize data to a value between 1 and 0
X_train = X_train/255
X_test = X_test/255

#flattens array from 28x28 to 784
num_pixels = 784
X_train = X_train.reshape(X_train.shape[0], num_pixels)
X_test = X_test.reshape(X_test.shape[0], num_pixels)

#creates model with keras
def create_model():
    model= Sequential()
    model.add(Dense(10, input_dim=num_pixels, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(Adam(0.01), loss="categorical_crossentropy", metrics=["accuracy"])
    return model
model = create_model()
print(model.summary())
history=model.fit(X_train, y_train, validation_split=0.1, epochs=30, batch_size=200, verbose=1, shuffle=1)

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

import requests
from PIL import Image

url = 'https://www.researchgate.net/profile/Jose_Sempere/publication/221258631/figure/fig1/AS:305526891139075@1449854695342/Handwritten-digit-2.png'
response = requests.get(url, stream=True)
img = Image.open(response.raw)
plt.imshow(img)
plt.show()

import cv2

img_array = np.asarray(img)
resized = cv2.resize(img_array, (28, 28))
gray_scale = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
image = cv2.bitwise_not(gray_scale)
plt.imshow(image, cmap=plt.get_cmap("gray"))
plt.show()

image = image/255
image = image.reshape(1, 784)
prediction = model.predict_classes(image)
print("predicted digit:", str(prediction))
