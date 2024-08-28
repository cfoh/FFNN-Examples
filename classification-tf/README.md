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
    model = Sequential()
    model.add(Dense(units=10, activation='relu', input_dim=len(X_train.columns)))
    model.add(Dense(units=20, activation='relu'))
    model.add(Dense(units=len(label_encoder.classes_), activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=24, epochs=100) 
```

We have 150 instances in the dataset, 80% (or 120) are used for training. Since each batch size is 24, the FFNN will be updated 5 times for each epoch, and we run 100 epochs. The following shows the final few epochs. In the final epoch, we achieve over 98% of accuracy.

```
...
Epoch 97/100
5/5 [==============================] - 0s 483us/step - loss: 0.2206 - accuracy: 0.9750
Epoch 98/100
5/5 [==============================] - 0s 748us/step - loss: 0.2177 - accuracy: 0.9750
Epoch 99/100
5/5 [==============================] - 0s 512us/step - loss: 0.2169 - accuracy: 0.9833
Epoch 100/100
5/5 [==============================] - 0s 477us/step - loss: 0.2118 - accuracy: 0.9833
```

![ffnn](https://github.com/cfoh/FFNN-Examples/assets/51439829/6a75fab2-e3e0-49c0-9eda-4f06d6959f11)
