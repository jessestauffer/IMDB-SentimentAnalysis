'''
	Input => Vectorized text
	Output => O if negative sentiment, 1 if positive sentiment
'''

from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt

VOCAB_SIZE = 10000

def decodeString(src):
	word_index = imdb.get_word_index()
	reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
	decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])	# replace words not in vocab with '?'
	return decoded_review

def vectorize_sequences(sequences, dimension=VOCAB_SIZE):
	results = np.zeros((len(sequences), dimension))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1.
	return results

if __name__ == "__main__":

	# load and pre-process data
	(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)	# only use most common 10,000 words
	X_train = vectorize_sequences(train_data)
	X_test = vectorize_sequences(test_data)
	y_train = np.asarray(train_labels).astype('float32')
	y_test = np.asarray(test_labels).astype('float32')

	# create validation set
	X_val = X_train[:10000]
	X_train = X_train[10000:]
	y_val = y_train[:10000]
	y_train = y_train[10000:]

	# build and train a model
	model = models.Sequential()
	model.add(layers.Dense(16, activation='relu', input_shape=(VOCAB_SIZE,)))
	model.add(layers.Dense(16, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))

	model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
	history = model.fit(X_train, y_train, epochs=4, batch_size=512, validation_data=(X_val, y_val))

	# plot training and validation loss
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs = range(1, len(acc) + 1)

	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()

	plt.plot(epochs, acc, 'bo', label='Training accuracy')
	plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
	plt.title('Accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.show()

	# evaluate the model on the test set
	evaluation = model.evaluate(X_test, y_test)
	print("Accuracy: " + str(evaluation[1]))
