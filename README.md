# Feed-Forward Neural Network (FFNN)

## Introduction

A feed-forward neural network (FFNN) is a fundamental architecture in machine learning (ML) and deep learning. It is a type of artificial neural network where the connections between nodes (neurons) do not form cycles, meaning that information flows in one direction, from the input layer to the output layer, without any loops or feedback connections.

This repo contains the following examples:
- [FFNN for Regression (single output)](https://github.com/cfoh/FFNN-Examples/tree/main/regression)
- [FFNN for Regression (multiple outputs)](https://github.com/cfoh/FFNN-Examples/tree/main/regression2)
- [FFNN for Classification](https://github.com/cfoh/FFNN-Examples/tree/main/classification)

#### Neural Network Structure

- Input Layer: The input layer consists of one or more neurons that receive the input data. 
  Each neuron represents a feature or attribute of the input data.
- Hidden Layers: Between the input and output layers, there can be one or more hidden layers. 
  Each hidden layer consists of multiple neurons (also called nodes or units). The term "hidden" comes from the fact that these layers are not directly connected to the input or output; their purpose is to perform intermediate computations.
- Output Layer: The output layer produces the final result or prediction. 
  The number of neurons in this layer depends on the type of problem. For binary classification problems, you might have one output neuron; for multi-class classification, there could be multiple output neurons, one for each class.

![ffnn](https://github.com/cfoh/FFNN-Examples/assets/51439829/52b8b1d8-6a36-4df4-a727-b73b092e164f)

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

## Setup Your Python Virtual Environment

Create a Python virtual environment for your project. This can be done by the following commands. After executing the first command shown below, you should see a new folder `venv` which contains the information about your virtual environment. The second command activates the environment.

```
python3 -m venv venv
source venv/bin/activate
```

Once your virtual environment is activated, you can install the packages and dependencies.

```
pip install numpy pandas scikit-learn tensorflow
```

## Explore the Examples

The repo contains several examples using FFNN to perform regression, function approximation and classification. They can be found in their individual folder. 

You can run the code by using the following command in the example folder. 

```
python ffnn.py
```

The program has 3 run modes:
- mode 1: train & save the model into a folder
- mode 2: train the model but don't save it
- mode 3: (default) load the saved model from the folder

By default, the program will read the model stored in the local folder and test the stored model. You can change the run mode to re-train the model.

