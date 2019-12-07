import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.utils import to_categorical
from keras import backend as K
import keras.callbacks
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

data = pd.read_csv('heart.csv')
print(data.describe)

#Clean data
nulls = pd.DataFrame(data.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns  = ['Null Count']
nulls.index.name  = 'Feature'
print(nulls)

data = data.dropna(axis=0)
data.hist(figsize = (12, 12))
plt.show(block=False)

#train data
x = np.array(data.drop(['target'], 1))
y = np.array(data['target'])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)

#normalize y variable
Ytrain = to_categorical(y_train, num_classes=None)
Ytest = to_categorical(y_test, num_classes=None)
print (Ytrain.shape)
print (Ytrain[:10])

#creating network
model = Sequential()
model.add(Dense(13, input_dim=13, activation='relu', kernel_initializer='normal'))
model.add(Dense(13, kernel_initializer='normal', activation='relu'))
model.add(Dense(2, activation='softmax'))

#compile model
epochs = 200
lrate = 0.001
adam = Adam(lr= lrate)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
print(model.summary())

#tensor board
tb = keras.callbacks.TensorBoard(log_dir=f".\logs\Tensors", histogram_freq=0,write_graph=True, write_images=True)

history=model.fit(x_train, Ytrain, validation_data=(x_test, Ytest),epochs=100, batch_size=5, verbose = 10, callbacks=[tb])

scores = model.evaluate(x_test, Ytest, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
print('Test  loss:', scores[0]*100)
model.save('./model' + '.h5')

# Model accuracy
# plt.figure(1)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model Accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'])
# plt.show(block=False)
#
# # Model Losss
# plt.figure(2)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'])
# plt.show()

#prediction
pred = np.argmax(model.predict(x_test), axis=1)

print('Results for Logistic Model')
print(accuracy_score(y_test, pred))
print(classification_report(y_test, pred))