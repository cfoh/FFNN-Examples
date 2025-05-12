'''
This is a simple CNN model to classify images of clock faces.

See the `train-12.py` file for the training code. The model is trained on a
reduced dataset with 12 classes (from 1:00 to 12:00). The model is saved as
`best_model.h5` in the `dataset-12` folder.

This program tests the model on a single image. 
'''

############################################################
# install the following dependencies:
# > pip install numpy tensorflow pillow
############################################################

from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import random, os

############################################################
# define dataset properties
# - image size is 224x224
# - number of classes is 12 (from 1:00 to 12:00)
############################################################

IMG = 224
num_of_classes = 12

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG, IMG), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to match training
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

############################################################
# load model
############################################################

best_model_file = "./dataset-12/best_model.h5"

model = load_model(best_model_file)

print(model.summary())

############################################################
# pick an image to test
############################################################

## pick a label to test
label = "08-00" # in the format "hh-mm"
folder = "./dataset-12/test/" + label

## list all files in the subfolder (filtering out non-image files if needed)
images = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

## randomly pick one
random_image = random.choice(images)
image_path = os.path.join(folder, random_image)

############################################################
# perform the prediction
############################################################

img_array = load_and_preprocess_image(image_path)
prediction = model.predict(img_array)
predicted_class_index = np.argmax(prediction)

img = Image.open(image_path)
img.show()

print(f"Predicted source: {image_path}")
print(f"Predicted time: {predicted_class_index+1}:00")
