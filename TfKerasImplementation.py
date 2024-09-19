import tensorflow as tf

# Load MNIST dataset
mnist = tf.keras.datasets.mnist 

"""
Print dataset details 28x28 images of hand-written digits 0-9
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("Training data shape:", train_images.shape)
print("Training labels shape:", train_labels.shape)
print("Test data shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)
"""

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(X_train,y_train, epochs= 3)

val_loss, vall_acc = model.evaluate(X_test,y_test)
print(('loss:',val_loss),('acc:', vall_acc))

import matplotlib.pyplot as plt

#plt.imshow(X_train[0], cmap = plt.cm.binary)
#plt.show()

#Normalization - scaling
import numpy as np

actual_label = y_test[10]
print('Actual label is equal to:', actual_label)

predictions = model.predict([X_test])
print('Prediction is equal to:', np.argmax(predictions[10]))




