from tensorflow.keras.preprocessing.image import ImageDataGenerator # Import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import numpy as np
import os
import zipfile
from google.colab import drive
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras
from keras.datasets import mnist
import numpy
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical

# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	print(trainX.shape)
	print(trainY.shape)
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# Image scaling
train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

train_dataset.class_indices
train_dataset.classes

# Function to display one image from each class
def visualize_samples(dataset, num_classes):
    plt.figure(figsize=(12, 6))

    # Get the class indices from the dataset
    class_indices = dataset.class_indices
    class_names = list(class_indices.keys())

    for i in range(num_classes):
        # Get one batch of images and labels
        images, labels = next(dataset)

        # Find the index of the current class
        class_index = i

        # Get the indices of images belonging to the current class
        indices = np.where(labels.argmax(axis=1) == class_index)[0]

        # Check if any images of the current class are present in the batch
        if len(indices) > 0:
            # Get the first image of the current class if available
            image = images[indices[0]]

            # Display the image
            plt.subplot(2, num_classes, i + 1)
            plt.imshow(image)
            plt.title(f'Train: {class_names[class_index]}')
            plt.axis('off')
        else:
            print(f"No images of class '{class_names[class_index]}' found in this batch.")

    plt.show()

# Visualize samples from the training dataset
#Import numpy
import numpy as np
visualize_samples(train_dataset, len(train_dataset.class_indices))

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	#print(max(train_norm))
	#print(min(train_norm))
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm


# define cnn model
# Import the library of regulizers
from keras import regularizers
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', padding= 'valid', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l1(0.01)))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	model.summary()
	plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

	return model

# creat a function that evaluates a model using the train and test datasets
# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=2):
	scores, histories = list(), list()
	# prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	for train_ix, test_ix in kfold.split(dataX):
		# define model
		model = define_model()
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model
		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
		# evaluate model
		acc = model.evaluate(testX, testY, verbose=0)
		#print('> %.3f' % (acc * 100.0))
		# stores scores
		scores.append(acc)
		histories.append(history)
	return scores, histories

# create a function that traces diagnostic learning curves
# plot diagnostic learning curves
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		pyplot.subplot(2, 1, 1)
		pyplot.title('Cross Entropy Loss')
		pyplot.plot(histories[i].history['loss'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		pyplot.subplot(2, 1, 2)
		pyplot.title('Classification Accuracy')
		pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	pyplot.show()
	

    # create a function that summarizes model performance
def summarize_performance(scores):
	scores = numpy.array(scores)
  #print scores
	Acc = scores[:,1]
	# print summary
	print('Accuracy: mean=%.3f std=%.3f'  % (mean(Acc)*100, std(Acc)*100))
	# box and whisker plots of results
	pyplot.boxplot(scores)
	pyplot.show()
	

# create a function that contains combines the different modules to test/evaluate the model
# run the different modules to test/evaluate the model
def run_test():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# evaluate model
	scores, histories = evaluate_model(trainX, trainY)
	# learning curves
	print(scores)
	summarize_diagnostics(histories)
	# summarize estimated performance
	summarize_performance(scores)

 # entry point, run the test harness
run_test()
