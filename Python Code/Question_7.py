# Reference https://towardsdatascience.com/how-to-generate-new-data-in-machine-learning-with-vae-variational-autoencoder-applied-to-mnist-ca68591acdcf
from keras import Sequential
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers
from keras.layers import Dense
from keras.utils import to_categorical
import keras

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# display the first image in the training data
plt.figure(1)
plt.imshow(train_images[0, :, :], cmap='gray')
plt.title('Ground Truth : {}'.format(train_labels[0]))
plt.show(block=False)

# process the data
# 1. convert each image of shape 28*28 to 784 dimensional which will be fed to the network as a single feature
dimData = np.prod(train_images.shape[1:])
train_data = train_images.reshape(train_images.shape[0], dimData)
test_data = test_images.reshape(test_images.shape[0], dimData)

# convert data to float and scale values between 0 and 1
train_data = train_data.astype('float')
test_data = test_data.astype('float')
# scale data
train_data /= 255.0
test_data /= 255.0

# change the labels frominteger to one-hot encoding
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

# Autoencoding - encoding and decoding an image
# For autoencoder model, we need to define an input image, encoding and then decoding an output image
input_image = layers.Input(shape=(dimData,))
encoded_dimension = 64
# Encoding the image to a smaller dimension
encoded_image = layers.Input(shape=(encoded_dimension,))

# adding multiple hidden layers
encoded_layer = Dense(encoded_dimension, activation='relu', input_shape=(dimData,))(input_image)
decoded_layer = Dense(dimData, activation='relu', input_shape=(dimData,))(encoded_layer)

# Encode model
model_encode = models.Model(input_image, encoded_layer)
print("Summary of Encoding model \n")
model_encode.summary()

# Autoencode model
model_autoencode = models.Model(input_image, decoded_layer)
print("Summary of Auto-encoding model \n")
model_autoencode.summary()

# Applying autoencoding layer to the input image
model_decode = models.Model(encoded_image, model_autoencode.layers[-1](encoded_image))
print("Summary of Decoding model \n")
model_decode.summary()

adadelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95)

model_autoencode.compile(optimizer=adadelta, loss='binary_crossentropy', metrics=['accuracy'])
history = model_autoencode.fit(train_data, train_data, batch_size=256, epochs=60, verbose=1,
                   validation_data=(test_data, test_data))

[test_loss, test_acc] = model_autoencode.evaluate(test_data, test_data)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
print(history.history.keys())

plt.figure(2)
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['loss'], label='Loss')
plt.legend(loc='upper left')
plt.show()
plt.figure(3)
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.plot(history.history['acc'], label='Accuracy')
plt.legend(loc='upper left')
plt.show()

plt.figure(4)
plt.imshow(test_images[0, :, :], cmap='gray')
plt.title('Test Image : {}'.format(test_labels[0]))
plt.show()

encoded_image_output = model_encode.predict(test_data)
decoded_image = model_decode.predict(encoded_image_output)

plt.figure(5)
plt.imshow(encoded_image_output[0].reshape((8, 8)), cmap='gray')
plt.title('Encoded Image :')
plt.show()

print(encoded_image_output[0, :])
plt.figure(6)
plt.imshow(decoded_image[0].reshape((28, 28)), cmap='gray')
plt.title('Decoded Image :')
plt.show()
