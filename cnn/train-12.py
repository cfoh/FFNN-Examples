'''
This is a simple CNN model to classify images of clock faces.

The original dataset is taken from 
https://www.kaggle.com/datasets/gpiosenka/time-image-datasetclassification
which contains images of clock faces with different times.
The dataset has 144 classes (from 1:00, 1:05, 1:10, ..., 12:55).
We have also created a reduced dataset with only 12 classes (from 1:00, 
2:00, ..., 12:00). Both datasets are available here, where `dataset-144` stores 
the original dataset and `dataset-12` stores the reduced dataset.

The dataset is divided into three folders: `train`, `test` and `valid`. Each of
these folders contains subfolders for each class. The images are stored in the
subfolders. The images are in RGB and have a size of 224x224 pixels. For each
class, there are 80 images in the `train` folder, 10 images in the `test` folder 
and 10 images in the `valid` folder.

This program trains a CNN model using the reduced dataset to classify the 
clock faces into 12 classes.
'''

############################################################
# install the following dependencies:
# > pip install numpy tensorflow pillow
############################################################

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import time

############################################################
# define dataset properties
# - image size is 224x224
# - number of classes is 12 (from 1:00 to 12:00)
############################################################

IMG = 224
num_of_classes = 12

############################################################
# set training parameters
############################################################

batch_size = 32
num_epochs = 15

############################################################
# define the model
############################################################

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG, IMG, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(num_of_classes, activation='softmax')

])

print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

############################################################
# load the dataset
############################################################

train_folder = "./dataset-12/train"  # training
test_folder = "./dataset-12/valid"   # validation

## training set
print("Loading training set...")
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.2)

training_set = train_datagen.flow_from_directory(train_folder,
                                              target_size=(IMG, IMG),
                                              color_mode='grayscale',
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              shuffle=True)

## the class labels are in the same order as the folders
## so expect to see class_label = {0: '01-00', 1: '02-00', ..., 11: '12-00'}
## which means that the predicted time is f"{predicted_class_index+1}:00"
class_labels = training_set.class_indices
class_labels = dict((v, k) for k, v in class_labels.items())  # invert dict
print("Labels:") 
print(class_labels) 

## testing set
print("Loading testing set...")
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_set = test_datagen.flow_from_directory(test_folder,
                                             target_size=(IMG, IMG),
                                             color_mode='grayscale',
                                             batch_size=batch_size,
                                             class_mode='categorical',
                                             shuffle=False)

############################################################
# train the model
############################################################

best_model_file = './dataset-12/best_model.h5'
best_model = ModelCheckpoint(
    filepath=best_model_file,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

t0 = time.time()
history = model.fit(training_set,
                    steps_per_epoch=int(np.ceil(training_set.samples/batch_size)),
                    epochs=num_epochs,
                    validation_data=test_set,
                    validation_steps=int(np.ceil(test_set.samples/batch_size)),
                    callbacks=[best_model],
                    verbose=1)

t1 = int(time.time() - t0)
print(f"Total train time: {str(t1)} seconds")
