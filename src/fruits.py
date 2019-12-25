import cv2
import keras
import os
import numpy as np
from time import time
from sys import argv
from keras.datasets import mnist
from keras.utils import to_categorical


def parse_args():
    epochs = int(argv[1])
    batch = int(argv[2])
    close = int(argv[3])
    learning_rate = float(argv[4])
    return epochs, batch, close, learning_rate


def load(path):
	images = []
	labels = []
	classes = os.listdir(path)
	for i in range(120):
		cur_path = path + classes[i] + '/'
		fruits = os.listdir(cur_path)
		for x in fruits:
			image = cv2.imread(cur_path + x)
			image = cv2.resize(image, (50, 50))
			images.append(image)
			labels.append(i)
	tensor = np.array(images)
	tensor = np.transpose(tensor, (0, 3, 1, 2))
	return tensor, labels


def main():
	start = time()
	epochs, batch, close, learning_rate = parse_args()
	train_images, train_labels = load('fruits-360_dataset/fruits-360/Training/')
	test_images, test_labels = load('fruits-360_dataset/fruits-360/Test/')     

	print('LOAD ' + str(time() - start))
	
	in_shape = train_images.shape
	print(in_shape)

	out_size = 50 * 50 * 3

	train_images = train_images.reshape((in_shape[0], out_size))
	train_images = train_images.astype('float32') / 255
	train_labels = to_categorical(train_labels)
	
	in_shape = test_images.shape
	test_images = test_images.reshape((in_shape[0], out_size))
	test_images = test_images.astype('float32') / 255
	test_labels = to_categorical(test_labels)
	
	num_classes = 120

	model = keras.Sequential([
        keras.layers.Dense(close, activation='relu', input_shape=(out_size,)),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

	sgd = keras.optimizers.SGD(lr=learning_rate)
	model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                
	start = time()
	model.fit(train_images, train_labels, epochs=epochs, batch_size=batch)
	train_time = time() - start

	test_loss, test_acc = model.evaluate(test_images, test_labels)
	print('\nTest accuracy:', test_acc)
	print('Time: ', train_time)


if __name__ == '__main__':
	main()
