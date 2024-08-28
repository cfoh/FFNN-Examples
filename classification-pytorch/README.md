# Feed-Forward Neural Network (FFNN) for Classification

## Introduction

This is a very simple example showing how FFNN can be used for classification.
We use a well-known [IRIS dataset](https://www.kaggle.com/datasets/uciml/iris) for classification.

![iris_img](https://github.com/cfoh/FFNN-Examples/assets/51439829/6453f36d-9831-4c02-9b8d-ef8e51e7aa46)
<p align="center">Image source: <a href="https://www.datacamp.com/tutorial/machine-learning-in-r">Machine Learning in R for beginners</a></p>

## The Dataset

The dataset is already given in `iris.data`. It's a csv file ready to use. The csv file contains 5 columns:
- sepal length in cm
- sepal width in cm
- petal length in cm
- petal width in cm
- class: 
  - Iris Setosa
  - Iris Versicolour
  - Iris Virginica

Here is a preview of the dataset. The first four columns are the input features, and the last column is the label identifying the type of Iris. Note that the csv file contains no header.

```
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
...
7.0,3.2,4.7,1.4,Iris-versicolor
6.4,3.2,4.5,1.5,Iris-versicolor
6.9,3.1,4.9,1.5,Iris-versicolor
5.5,2.3,4.0,1.3,Iris-versicolor
...
6.3,3.3,6.0,2.5,Iris-virginica
5.8,2.7,5.1,1.9,Iris-virginica
7.1,3.0,5.9,2.1,Iris-virginica
6.3,2.9,5.6,1.8,Iris-virginica
...
```

## FFNN Design

The following is the default FFNN design in the code:

```python
  class FFNN(nn.Module):
      def __init__(self, input_dim, output_dim):
          super(FFNN, self).__init__()
          self.fc1 = nn.Linear(input_dim, 10)
          self.fc2 = nn.Linear(10, 20)
          self.fc3 = nn.Linear(20, output_dim)
      
      def forward(self, x):
          x = torch.relu(self.fc1(x))
          x = torch.relu(self.fc2(x))
          x = torch.softmax(self.fc3(x), dim=1)
          return x
```

We have 150 instances in the dataset, 80% (or 120) are used for training. Since each batch size is 24, the FFNN will be updated 5 times for each epoch, and we run 100 epochs. The following shows the final few epochs. In the final epoch, we achieve over 98% of accuracy.

```
...
Epoch 92/100, Loss: 0.6439, Accuracy: 0.9667
Epoch 93/100, Loss: 0.6424, Accuracy: 0.9667
Epoch 94/100, Loss: 0.6414, Accuracy: 0.9750
Epoch 95/100, Loss: 0.6394, Accuracy: 0.9750
Epoch 96/100, Loss: 0.6384, Accuracy: 0.9750
Epoch 97/100, Loss: 0.6372, Accuracy: 0.9750
Epoch 98/100, Loss: 0.6355, Accuracy: 0.9750
Epoch 99/100, Loss: 0.6345, Accuracy: 0.9750
Epoch 100/100, Loss: 0.6346, Accuracy: 0.9667
```

The following is the outcome using the provided stored model:

```
Loading stored model...

Model performance:
> Test Loss: 0.646
> Test Accuracy: 0.900

NN_output = [9.6192125e-06 7.8912191e-02 9.2107815e-01]
             0.00001 prob for Iris-Setosa
             0.07891 prob for Iris-Versicolour
             0.92108 prob for Iris Virginica
Prediction   = Iris-virginica
Actual_Label = Iris-virginica
```

![ffnn](https://github.com/cfoh/FFNN-Examples/assets/51439829/6a75fab2-e3e0-49c0-9eda-4f06d6959f11)
