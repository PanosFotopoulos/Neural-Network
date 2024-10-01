import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

CATEGORIES = ['Dog', 'Cat']

def prepare(filepath):
    IMG_SIZE = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Convert the image to grayscale
    if img_array is None:
        raise FileNotFoundError(f"Unable to load image at path: {filepath}")
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

image_path = r"C:\Users\Panos\Desktop\pred\shiera\shiera5.png"

try:
    image_data = prepare(image_path)

    model = load_model(r'C:\Users\Panos\Desktop\PythonProject\n_n\Final_Models\3x32x1-CNN1.keras') 

    prediction = model.predict(image_data)
    print('The type of the animal in the picture is a', CATEGORIES[int(prediction[0][0])])

except FileNotFoundError as e:
    print(e)
