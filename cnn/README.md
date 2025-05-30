# Convolutional Neural Network (CNN)

## 1. Introduction

### 1.1 Reading Time from Clock Faces: A CNN Approach

In cities around the world, clock towers and large clocks are not only architectural features but also iconic symbols. These clocks come in many forms—round or square faces, Roman or Arabic numerals, vibrant colors or muted tones, intricate designs or minimalist styles. For humans, reading time from such clocks is often second nature. However, for machines, this seemingly simple task presents a unique challenge.

### 1.2 Why is this challenging?

While identifying the presence of a clock might be manageable using basic object detection techniques, accurately reading the time from diverse clock faces is a far more complex problem. Variations in clock design—ranging from differences in hand styles, color contrasts, background clutter, and even weather conditions—make it difficult to generalize a model to accurately interpret time from a wide range of clocks in real-world environments. This research work entitled ["It’s About Time: Analog Clock Reading in the Wild"](https://arxiv.org/pdf/2111.09162) published in CVPR 2022 attempts to tackle this challenge.

In this project, we focus on simplifying this challenge by training a Convolutional Neural Network (CNN) model to recognize and classify clock face images based on the time they display. Instead of reading the time as continuous values, we frame the problem as a classification task, where each class corresponds to a specific time on the clock. For instance, one class may represent 1:00, another for 2:00, and so on, up to 12:00.

## 2. The Dataset

To train our model, we use a dataset of clock face images sourced from Kaggle (https://www.kaggle.com/datasets/gpiosenka/time-image-datasetclassification). The original dataset contains images representing 144 distinct times (e.g., 1:00, 1:05, 1:10, ..., up to 12:55). For simplicity and clarity, we also create a reduced dataset with only 12 classes, each corresponding to the exact hour (from 1:00 to 12:00). This reduction allows us to focus the model on reading broad time categories while managing the complexity of variations in clock designs.

The dataset is structured into `train`, `test`, and `valid` folders, with each class having its own subfolder containing images. Each image is in RGB format and resized to 224x224 pixels for input into the model. In the reduced dataset, for each class, we include 80 images in the training set, 10 in the validation set, and 10 in the test set.

The following shows some training images from the dataset with a label of `04-00`:

<img src="https://github.com/cfoh/FFNN-Examples/blob/main/cnn/dataset-12/train/04-00/10.jpg" alt="4pm" style="width:50pt"><img src="https://github.com/cfoh/FFNN-Examples/blob/main/cnn/dataset-12/train/04-00/12.jpg" alt="4pm" style="width:50pt"><img src="https://github.com/cfoh/FFNN-Examples/blob/main/cnn/dataset-12/train/04-00/15.jpg" alt="4pm" style="width:50pt"><img src="https://github.com/cfoh/FFNN-Examples/blob/main/cnn/dataset-12/train/04-00/20.jpg" alt="4pm" style="width:50pt"><img src="https://github.com/cfoh/FFNN-Examples/blob/main/cnn/dataset-12/train/04-00/22.jpg" alt="4pm" style="width:50pt"><img src="https://github.com/cfoh/FFNN-Examples/blob/main/cnn/dataset-12/train/04-00/27.jpg" alt="4pm" style="width:50pt"><img src="https://github.com/cfoh/FFNN-Examples/blob/main/cnn/dataset-12/train/04-00/32.jpg" alt="4pm" style="width:50pt">


## 3. CNN Design

Using TensorFlow, we design a CNN architecture that can extract visual features from the clock faces. The convolutional layers help the model learn patterns such as hand positions and relative angles, while pooling layers reduce the dimensionality of these feature maps. The fully connected layers then map these patterns to one of the 12 output classes. The model is trained using a categorical cross-entropy loss function, with an optimizer like Adam to adjust weights and minimize classification errors.

By training this CNN model, we aim to equip machines with the ability to not just detect clocks, but to accurately "read" the time they display—even when those clocks vary widely in design and appearance. This approach offers a foundation for further exploration into more advanced time-reading models that could handle continuous time estimation, complex clock styles, and challenging real-world environments.

```python
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
```

![Illustration of the CNN Structure](https://github.com/cfoh/FFNN-Examples/blob/main/cnn/cnn_diagram.svg)

Model explanation:
- **Input Layer**: The model takes grayscale clock face images resized to (IMG, IMG, 1), where IMG is typically 224 pixels.
- **Convolutional Layers**: There are three convolutional layers with 32 and 64 filters, each using a 3x3 kernel and ReLU activation to learn spatial features.
- **Pooling Layers**: MaxPooling layers with a 2x2 pool size reduce spatial dimensions and help in abstraction.
- **Flattening Layer**: Converts the 3D feature maps into a 1D vector.
- **Dropout Layer**: Introduces a dropout rate of 0.5 to reduce overfitting during training.
- **Dense Layers**: A fully connected layer with 512 neurons and ReLU activation captures complex patterns.
- **Output Layer**: The final dense layer uses softmax activation to output probabilities across num_of_classes (12 in our case) representing the possible clock times.

## 4. Executing the Example

You can run the example in two ways:

### 4.1 Run it locally

Clone the repository, create corresponding virtual environment, then run the following on the terminal:
- `python train-12.py` to train the save the best model inside `dataset-12` folder, and
- `python test-12.py` to test the model with one random instance from the `test` folder.

### 4.2 Run it on Google Colab

Do the following:
- Download the following files from this repository:
  - `copy-dataset.ipynb`
  - `train-12.ipynb`
  - `test-12.ipynb`
- Create a folder called `lesson` in your Google Drive.
- Upload `copy-dataset.ipynb` to your Google Drive under `lesson`.
  - Note that if you use a different folder name, you need to change the name
    in `copy-dataset.ipynb` accordingly.
- Run `copy-dataset.ipynb` which will (i) copy our shared `dataset-12.zip` dataset file
  to your Google Drive folder, and (ii) unzip it to create the dataset folders in your folder.
- Upload `train-12.ipynb` and `test-12.ipynb` to your Google Drive under `lesson`.
- Run `train-12.ipynb` to train the save the best model inside `dataset-12` folder.
- Once the best model is saved, you can run `test-12.ipynb` to test the model with 
  one random instance from the `test` folder.
