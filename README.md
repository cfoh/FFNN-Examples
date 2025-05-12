# Feed-Forward Neural Network (FFNN) ![Machine Learning: FFNN](https://img.shields.io/badge/Machine%20Learning-Neural%20Network-blueviolet) ![MIT License](https://img.shields.io/badge/License-MIT-green?logo=github)

## Table of Contents
- [1. Introduction](#intro)
  - [1.1 Simple 1D Feed-Forward Neural Network Structure](#simple-FFNN)
  - [1.2 Convolutional Neural Network (CNN)](#CNN)
- [2. Setup Your Python Virtual Environment](#setup)
- [3. Explore simple 1D FFNN Examples](#simple-examples)
  - FFNN for Regression using (single output)
    - [Using Tensorflow](https://github.com/cfoh/FFNN-Examples/tree/main/regression-tf)
    - [Using Pytorch](https://github.com/cfoh/FFNN-Examples/tree/main/regression-pytorch)
  - FFNN for Regression using Tensorflow (multiple outputs)
    - [Using Tensorflow](https://github.com/cfoh/FFNN-Examples/tree/main/regression-tf2)
    - [Using Pytorch](https://github.com/cfoh/FFNN-Examples/tree/main/regression-pytorch2)  
  - FFNN for Classification
    - [Using Tensorflow](https://github.com/cfoh/FFNN-Examples/tree/main/classification-tf)
    - [Using Pytorch](https://github.com/cfoh/FFNN-Examples/tree/main/classification-pytorch)

## 1. Introduction <a name=intro></a>

A feed-forward neural network (FFNN) is a fundamental architecture in machine learning (ML) and deep learning. It is a type of artificial neural network where the connections between nodes (neurons) do not form cycles, meaning that information flows in one direction, from the input layer to the output layer, without any loops or feedback connections.

We can classify FFNN based on its structure. The commonly used FFNNs are:
- simple FFNN with 1-dimensional (1D) inputs
- convolutional neural network (CNND) with 2-dimensional (2D) inputs

### 1.1 Simple 1D Feed-Forward Neural Network Structure <a name=simple-FFNN></a>

- Input Layer: The input layer consists of one or more neurons that receive the input data. 
  Each neuron represents a feature or attribute of the input data.
- Hidden Layers: Between the input and output layers, there can be one or more hidden layers. 
  Each hidden layer consists of multiple neurons (also called nodes or units). The term "hidden" comes from the fact that these layers are not directly connected to the input or output; their purpose is to perform intermediate computations.
- Output Layer: The output layer produces the final result or prediction. 
  The number of neurons in this layer depends on the type of problem. For binary classification problems, you might have one output neuron; for multi-class classification, there could be multiple output neurons, one for each class.

The following diagram shows a simple feed-forward neural network structure:

![ffnn](https://github.com/cfoh/FFNN-Examples/assets/51439829/f1ca2896-5c9a-45f6-8473-3cc727c5ff37)

#### 1.1.1 Feed-forward Process

During the feed-forward process, data flows through the network in one direction, from the input layer to the output layer.
Each neuron in a hidden or output layer computes a weighted sum of its inputs, applies an activation function to this sum, and passes the result as output to the next layer.
The weighted sum calculation involves multiplying each input by a corresponding weight and summing these products, often adding a bias term.
The activation function introduces non-linearity into the model, allowing the network to capture complex patterns in the data. Common activation functions include ReLU (Rectified Linear Unit), sigmoid, and tanh.

Given a trained FFNN model, this process performs **inference** where it computes the output(s) based on the given inputs.

#### 1.1.2 Training

To make the FFNN model learn from data, you need a labeled dataset and a loss function that quantifies how well the model's predictions match the actual labels.

Optimization algorithms like gradient descent are used to minimize the loss by adjusting the weights and biases in the FFNN model.

Backpropagation is a key technique for updating the model's parameters. It computes the gradients of the loss with respect to the weights of the FFNN and propagates these gradients backward through the layers to make weight updates.

#### 1.1.3 Applications

FFNNs are versatile and can be used for various machine learning tasks, including classification, regression, and function approximation.

#### 1.1.4 Challenges

The architecture and performance of an FFNN depend on hyperparameters like the number of layers, the number of neurons in each layer, and the choice of activation functions.

Overfitting is a common challenge; regularization techniques like dropout or L2 regularization are often applied to mitigate this issue.

### 1.2 Convolutional Neural Network (CNN) <a name=CNN></a>

A CNN is a specialized type of feed-forward neural network designed to process data with a grid-like topology or 2D inputs, such as images. Unlike traditional neural networks that use fully connected layers, CNNs use convolutional layers to automatically and efficiently extract local features by applying filters (kernels) that scan across the input. This architecture allows CNNs to capture spatial hierarchies and patterns, such as edges, textures, and object parts, in a way that is translation-invariant. CNNs often include pooling layers to reduce dimensionality and computational complexity, followed by fully connected layers for classification or regression. They are widely used in computer vision tasks like image recognition, object detection, and medical imaging due to their high accuracy and ability to learn directly from raw data.

The following shows an example of a well-known FFNN called LeNet-5 which includes a series of CNN architectures.

![LeNet-5](https://github.com/user-attachments/assets/b72cea8e-0999-4e75-88ba-bd71b1923746)

#### 1.2.1 The Convolution and Pooling Operations

As shown in LeNet-5, apart from the dense layer, two new layers are introduced, which are **convolutional** and **pooling** layers. The corresponding operations of convolutional and pooling layers are **convolution** and **pooling** operations respectively. An excellent explanation of these two operations can be found in this tutorial article: [A guide to convolution arithmetic for deep learning](https://arxiv.org/pdf/1603.07285).

In **convolution operation**, a filter (or kernel), $K$ is used to perform convolution with the input image $I$ to produce the output $S$. The following illustrates the convolution operation based on the example given in Fig. 1.1 of the tutorial article where a 3-by-3 filter (shown in subscripts) is used to apply on a 5-by-5 image (shown in blue) to produce a 3-by-3 output (shown in green). The convolution operation essentially performs a weighted sum by applying an element-wise product between the filter and a local region of the input image:

$$S(i, j) = \sum_{m} \sum_{n} I(i + m, j + n) \cdot K(m, n)$$

<img src="https://github.com/user-attachments/assets/ac84911f-6c94-4b03-bc53-6db29c179d14" width="400"/>

Likewise, in **max pooling operation**, a sliding window, $\Omega$ is used to perform max pooling on the input $x$ to produce the output $y$. The following illustrates the convolution operation based on the example given in Fig. 1.6 of the tutorial article where a 3-by-3 window (shaded area) is used to apply on a 5-by-5 input (shown in blue) to produce a 3-by-3 output (shown in green). As can be seen, in each step, the output is simply the maximum value of the local region of $x$ covered by the sliding window.

$$y(i, j) = \max_{(m, n) \in \Omega(i,j)} x(m, n)$$

<img src="https://github.com/user-attachments/assets/ce9c962c-6a10-4107-9442-fb224b50d78f" width="400"/>

#### 1.2.2 Applications

CNNs are widely applied in tasks that involve spatial or visual data due to their ability to automatically learn and extract meaningful features from raw inputs. In computer vision, CNNs power applications such as image classification, object detection, facial recognition, and image segmentation.

## 2. Setup Your Python Virtual Environment <a name=setup></a>

Create a Python virtual environment for your project. This can be done by the following commands. After executing the command shown below, you should see a new folder `venv` which contains the information about your virtual environment. 

```
python3 -m venv venv
```

All examples can share the same virtual environment. To achieve this, you can create `venv` under the parent folder. When trying with an example, say `regression-tf`, you should change directory to the example folder, and then activate the virtual environment by providing the its path:

```
cd regression-tf
source ../venv/bin/activate
```

Once your virtual environment is activated, you can install the packages and dependencies.

```
pip install numpy pandas scikit-learn 
pip install tensorflow
pip install torch
```

## 3. Explore simple 1D FFNN Examples <a name=simple-examples></a>

The repo contains several examples using FFNN to perform regression, function approximation and classification. They can be found in their individual folder. 

You can run the code by using the following command in the example folder. 

```
python ffnn.py
```

The program has 3 run modes:
- mode 1: train & save the model into a folder
- mode 2: train the model but don't save it
- mode 3: (default) load the saved model from the folder

By default, the program will read the model stored in the local folder and test the stored model. You can change the run mode to `mode 1` to re-train the model.
