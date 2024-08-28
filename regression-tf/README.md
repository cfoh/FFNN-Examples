# FFNN for Regression (Single Output)

## Introduction

This is a very simple example showing how FFNN can be used for regression.
In this example, the function for the FFNN to predict is 
$$f(x_1,x_2) = x_2 sin(x_1) - x_1 cos(x_2)$$ 
where $x_1$ and $x_2$ are two inputs between -5.0 and 5.0. 

The following is the plot of the function:

![sample_plot](https://github.com/cfoh/FFNN-Examples/assets/51439829/6aa48810-596e-4cd3-b0b0-098b832c07b0)

## The Dataset

The dataset is already given in `sample_data.csv`. It's ready to use. The csv file will be loaded in the program for training and testing.

You can also regenerate a new set of dataset by using `sample_generator.xlsx`. You can simply open and touch the file to generate new random inputs in the spreadsheet, then save the file as `sample_data.csv`. 

Here is a preview of the dataset:

```
x1,x2,y
-4.1013795466878,3.28250953731546,-1.37212229345511
-3.11159144717724,0.637153839007517,2.48196048442536
4.54208942617528,-0.304253799524342,-4.03362226034554
4.65949379148264,-1.38263933701234,0.509153141224465
-4.97952272924988,0.543734582772977,4.78583812190707
1.69711052031977,0.572528129239565,-0.858512362319969
...
```

## FFNN Design

The following is the default FFNN design in the code:

```python
    model = Sequential()
    model.add(Dense(units=6, activation='relu', input_dim=len(X_train.columns)))
    model.add(Dense(units=12, activation='tanh'))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(X_train, y_train, batch_size=50, epochs=200)
```

We have 5000 instances in our dataset, 80% (or 4000) are used for training. Since each batch size is 50, the FFNN will be updated 80 times for each epoch, and we run 200 epochs. The following shows the final few epochs. In the final epoch, we achieve a loss of below 0.24.

```
...
Epoch 197/200
80/80 [==============================] - 0s 657us/step - loss: 0.2446
Epoch 198/200
80/80 [==============================] - 0s 646us/step - loss: 0.2423
Epoch 199/200
80/80 [==============================] - 0s 624us/step - loss: 0.2428
Epoch 200/200
80/80 [==============================] - 0s 639us/step - loss: 0.2383
```

The following is the outcome using the provided stored model:

```
Loading stored model...

Model performance:
> Loss on test set: 0.181

More test results using random inputs:
> x1,x2 = -0.86, +0.14; predicted = +0.74; actual=+0.74; diff=0.00
> x1,x2 = +1.52, +3.46; predicted = +4.65; actual=+4.90; diff=0.25
> x1,x2 = +0.41, -1.55; predicted = -1.20; actual=-0.63; diff=0.57
> x1,x2 = -1.55, +0.84; predicted = +0.66; actual=+0.19; diff=0.47
> x1,x2 = -0.17, +3.57; predicted = -0.59; actual=-0.76; diff=0.18
```

The model is now ready for further prediction. Do the following to predict a single point:

```python
x1, x2 = -0.8, 3.6
y = model.predict(np.array([[x1,x2]]),verbose=0)[0][0]
```

![ffnn](https://github.com/cfoh/FFNN-Examples/assets/51439829/838c3a4a-7951-4bb9-a187-c2ba3d69fb62)
