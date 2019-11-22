# Dataset https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data
from keras.models import Sequential
from keras import layers
import keras
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.layers import Dropout,Dense
from keras.layers import Conv1D,MaxPooling1D
import matplotlib.pyplot as plt

df_train = pd.read_csv('sentiment-analysis-on-movie-reviews/train.tsv', encoding='latin-1', sep='\t')
df_test = pd.read_csv('sentiment-analysis-on-movie-reviews/test.tsv', encoding='latin-1', sep='\t')

# df = df.select_dtypes(include=[np.number]).interpolate().dropna()
print(df_train.head())
print(df_test.head())

sentences_train = df_train.Phrase
y_train = df_train.Sentiment
y_train = keras.utils.to_categorical(y_train.values)
# test data tsv
sentences_test = df_test.Phrase

# Splitting data into train and validation

X_train, X_val, y_train, y_val = train_test_split(sentences_train, y_train, test_size=0.2, random_state=1000)

# tokenizing data
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(X_train.values)
vocab_size = len(tokenizer.word_index) + 1

# Embedding
max_review_len = max([len(s.split()) for s in X_train.values])
print(max_review_len)
vocab_size = len(tokenizer.word_index) + 1
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(sentences_test)

X_train = sequence.pad_sequences(X_train, maxlen=max_review_len)
X_val = sequence.pad_sequences(X_val, maxlen=max_review_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_len)


# Number of features
# print(input_dim)
model = Sequential()
# Add Embedding
model.add(Embedding(vocab_size, 32, input_length=max_review_len))
model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())

# Output
model.add(Dense(5, activation='softmax'))
lrate = 0.01
epochs = 25
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
adam = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=epochs, verbose=True, validation_data=(X_val, y_val), batch_size=256)

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


