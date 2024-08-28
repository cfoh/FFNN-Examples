# FFNN for Regression (Multiple Outputs)

## Introduction

This is a very simple example showing how FFNN can be used for predicting a function with two outputs.
In this example, the function for the FFNN to predict is a complex function that takes two inputs (i.e. real and imaginary) and produces two outputs (i.e. real and imaginary).
$$f(z) = z^2 + 2z - j$$ 
where $z$ is the complex input with values between -2.0 and 2.0 for both real and imaginary parts, and $j$ is the imaginary unit.

The following is the plot of the function:

![sample_plot](https://github.com/cfoh/FFNN-Examples/assets/51439829/f7c20e9e-fb8b-41db-86da-6847fc67185a)

## The Dataset

The dataset is already given in `sample_data.csv`. It's ready to use. The csv file will be loaded in the program for training and testing.

You can also regenerate a new set of dataset by using `sample_generator.py`. The program will also plot the function.

Here is a preview of the dataset. The first two columns are the inputs, and the last two columns are the outputs.

```
Real(z),Imag(z),Real(f(z)),Imag(f(z))
1.02951146000298,0.3025450353583081,3.0273832678634673,0.2280372328533864
-0.6619820835484402,-0.5311632796696606,-1.1678783178271765,-1.3590854101790315
-0.5890470006960635,-0.3086021023897061,-0.9263528899624454,-1.2536419191371004
-0.7449535482372025,-0.6113209011230176,-1.3086645515930653,-1.3118304534397232
-1.3836807383984695,-1.0650743958746078,-1.987172559729666,-0.18270293868305343
-0.5835490375046413,-0.7766605183792925,-1.4297701566458807,-1.6468820408224012
...
```

## FFNN Design

The following is the default FFNN design in the code:

```python
    model = Sequential()
    model.add(Dense(units=6, activation='relu', input_dim=len(X_train.columns)))
    model.add(Dense(units=12, activation='tanh'))
    model.add(Dense(units=len(y_train.columns)))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(X_train, y_train, batch_size=50, epochs=200)
```

We have 5000 instances in our dataset, 80% (or 4000) are used for training. Since each batch size is 50, the FFNN will be updated 80 times for each epoch, and we run 200 epochs. The following shows the final few epochs. In the final epoch, we achieve a loss of around 0.0165.

```
...
Epoch 197/200
80/80 [==============================] - 0s 560us/step - loss: 0.0167
Epoch 198/200
80/80 [==============================] - 0s 607us/step - loss: 0.0166
Epoch 199/200
80/80 [==============================] - 0s 601us/step - loss: 0.0166
Epoch 200/200
80/80 [==============================] - 0s 587us/step - loss: 0.0165
```

The model is now ready for prediction. Do the following to predict a single point:

```python
z_real, z_imag = -0.8, 1.6
f_real, f_imag = model.predict(np.array([[z_real,z_imag]]),verbose=0)[0]
```

![ffnn](https://github.com/cfoh/FFNN-Examples/assets/51439829/3dc94771-da56-4f01-93fc-c4e2ab3e76c0)
