# Dataset from https://www.kaggle.com/ronitf/heart-disease-uci
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from keras.optimizers import SGD, Nadam, rmsprop
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras.datasets import boston_housing
from keras.layers.core import Dropout
from keras import regularizers
from keras.callbacks import TensorBoard
from datetime import datetime
import numpy as np

df = pd.read_csv('heart.csv', encoding='latin-1')

# df = df.select_dtypes(include=[np.number]).interpolate().dropna()
print(df.head())
y = df.target.values
x = df.drop(['target'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
x_train_scale = scaler.transform(X_train)
x_test_scale = scaler.transform(X_test)
# Number of features
# print(input_dim)
model = Sequential()
DROPOUT = 0.2
#correct input_dim
model.add(layers.Dense(16,input_dim=13, kernel_initializer='normal', activation='sigmoid'))
model.add(layers.Dense(32,input_dim=13, kernel_initializer='normal', activation='sigmoid'))
# model.add(layers.Dense(64,input_dim=13, kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(DROPOUT))
#change the output to softmax
model.add(layers.Dense(1))

sgd = SGD(lr=0.002, momentum=0.9, nesterov=False)
adam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
Rmsprop = rmsprop(lr=0.002)
model.compile(loss='mean_squared_error', optimizer=Rmsprop,metrics=['acc'])

tensorboard = TensorBoard(log_dir=f".\logs\Tensors")
history = model.fit(x_train_scale, y_train, epochs=100, verbose=True, validation_data=(x_test_scale, y_test), batch_size=8, callbacks=[tensorboard])

[test_loss, test_acc] = model.evaluate(x_test_scale, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
print(history.history.keys())

plt.figure(1)
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['loss'], label='Loss')
plt.legend(loc='upper left')
plt.show()
plt.figure(2)
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.plot(history.history['acc'], label='Accuracy')
plt.legend(loc='upper left')
plt.show()