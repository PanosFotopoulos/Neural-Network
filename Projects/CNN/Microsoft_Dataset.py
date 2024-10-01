import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIER = r"C:\Users\Panos\Desktop\cats_and_dogs\PetImages"
CATEGORIES = ["Cat","Dog" ]
IMG_SIZE = 50

training_data = []

def create_training_data():
    for catergory in CATEGORIES:
        path = os.path.join(DATADIER, catergory)# gives the path to the respective dir
        class_num = CATEGORIES.index(catergory)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE)) # 50X50
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
                
create_training_data()

   
random.shuffle(training_data)

X = []
y = []

for features, labels in training_data:
    X.append(features)
    y.append(labels)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

with open("X.pickle", "wb") as pickle_out:
    pickle.dump(X, pickle_out)

with open("y.pickle", "wb") as pickle_out:
    pickle.dump(y, pickle_out)

