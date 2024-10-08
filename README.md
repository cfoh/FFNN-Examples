# Feed-Forward Neural Network (FFNN) ![Machine Learning: FFNN](https://img.shields.io/badge/Machine%20Learning-Neural%20Network-blueviolet) ![MIT License](https://img.shields.io/badge/License-MIT-green?logo=github)

## Table of Contents
- [Introduction](#intro)
- [Setup Your Python Virtual Environment](#setup)
- [Explore the Examples](#examples)
  - FFNN for Regression using (single output)
    - [Using Tensorflow](https://github.com/cfoh/FFNN-Examples/tree/main/regression-tf)
    - [Using Pytorch](https://github.com/cfoh/FFNN-Examples/tree/main/regression-pytorch)
  - FFNN for Regression using Tensorflow (multiple outputs)
    - [Using Tensorflow](https://github.com/cfoh/FFNN-Examples/tree/main/regression-tf2)
    - [Using Pytorch](https://github.com/cfoh/FFNN-Examples/tree/main/regression-pytorch2)  
  - FFNN for Classification
    - [Using Tensorflow](https://github.com/cfoh/FFNN-Examples/tree/main/classification-tf)
    - [Using Pytorch](https://github.com/cfoh/FFNN-Examples/tree/main/classification-pytorch)

## Introduction <a name=intro></a>

A feed-forward neural network (FFNN) is a fundamental architecture in machine learning (ML) and deep learning. It is a type of artificial neural network where the connections between nodes (neurons) do not form cycles, meaning that information flows in one direction, from the input layer to the output layer, without any loops or feedback connections.

#### Neural Network Structure

- Input Layer: The input layer consists of one or more neurons that receive the input data. 
  Each neuron represents a feature or attribute of the input data.
- Hidden Layers: Between the input and output layers, there can be one or more hidden layers. 
  Each hidden layer consists of multiple neurons (also called nodes or units). The term "hidden" comes from the fact that these layers are not directly connected to the input or output; their purpose is to perform intermediate computations.
- Output Layer: The output layer produces the final result or prediction. 
  The number of neurons in this layer depends on the type of problem. For binary classification problems, you might have one output neuron; for multi-class classification, there could be multiple output neurons, one for each class.

![ffnn](https://github.com/cfoh/FFNN-Examples/assets/51439829/f1ca2896-5c9a-45f6-8473-3cc727c5ff37)


#### Feed-forward Process

During the feed-forward process, data flows through the network in one direction, from the input layer to the output layer.
Each neuron in a hidden or output layer computes a weighted sum of its inputs, applies an activation function to this sum, and passes the result as output to the next layer.
The weighted sum calculation involves multiplying each input by a corresponding weight and summing these products, often adding a bias term.
The activation function introduces non-linearity into the model, allowing the network to capture complex patterns in the data. Common activation functions include ReLU (Rectified Linear Unit), sigmoid, and tanh.

#### Training

To make the network learn from data, you need a labeled dataset and a loss function that quantifies how well the network's predictions match the actual labels.

Optimization algorithms like gradient descent are used to minimize the loss by adjusting the weights and biases in the network.

Backpropagation is a key technique for updating the network's parameters. It computes the gradients of the loss with respect to the network's weights and propagates these gradients backward through the layers to make weight updates.

#### Applications

FFNNs are versatile and can be used for various machine learning tasks, including classification, regression, and function approximation.

#### Challenges

The architecture and performance of an FFNN depend on hyperparameters like the number of layers, the number of neurons in each layer, and the choice of activation functions.

Overfitting is a common challenge; regularization techniques like dropout or L2 regularization are often applied to mitigate this issue.

## Setup Your Python Virtual Environment <a name=setup></a>

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

## Explore the Examples <a name=examples></a>

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
