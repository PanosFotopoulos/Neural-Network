import numpy as np
import pickle
import time
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

#NAME = "Cats-vs-Dogs-cnn-64x2-{}".format(str(time.time()))

#tensorboard = TensorBoard(log_dir='./logs/{}'.format(NAME))

# Load the preprocessed data from pickle files
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))


# Debugging: Check the shape and data types
print("Shape of X:", X.shape)
print("Type of X:", type(X))
print("First element type of X:", type(X[0][0][0][0]))  # Check an element inside X to see if it's numeric

print("Length of y:", len(y))
print("Type of y:", type(y))
print("First element of y:", type(y[0])) 

# Normalize the pixel values (X should already be a NumPy array)
X = X / 255.0


import time
dense_layers = [0, 1, 2]
layer_sizes  = [32, 64, 128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = '{}-conv-{}-nodes-{}-dense-{}'.format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir='./model_logs/{}'.format(NAME))
            # Model setup
            model = Sequential()
            # 1st Convolution layer
            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            # 2nd Convolution layer
            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            
            
            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            # Output layer
            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            # Compile the model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Train the model
            model.fit(X, np.array(y), batch_size=32, epochs=10, validation_split=0.1,callbacks=[tensorboard])#testorboard --logdir='logs/'

