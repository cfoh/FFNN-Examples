############################################################
# install the following dependencies:
# > pip install numpy pandas scikit-learn tensorflow
############################################################

import pandas as pd
import numpy as np
import math, random
from sklearn.model_selection import train_test_split

############################################################
# the csv specifies the function to predict,
# it contains 3 columns:
# - column 1: `x1`, float between -5.0 & 5.0
# - column 2: `x2`, float between -5.0 & 5.0
# - column 3: `y`,  float y = x2*sin(x1) - x1*cos(x2)
############################################################

df = pd.read_csv("sample_data.csv")

X = df[["x1","x2"]]   # the input columns
y = df[["y"]]         # the output column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

############################################################
# training
############################################################

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

run_mode = 3    # 1: train & save the model into a folder
                # 2: train but don't save the model
                # 3: load the saved model

if run_mode==1 or run_mode==2:
    model = Sequential()

    model.add(Dense(units=6, activation='relu', input_dim=len(X_train.columns)))
    model.add(Dense(units=12, activation='tanh'))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(X_train, y_train, batch_size=50, epochs=200)  # loss: 0.17

    # model.add(Dense(units=6, activation='relu', input_dim=len(X_train.columns)))
    # model.add(Dense(units=12, activation='tanh'))
    # model.add(Dense(units=1))
    # model.compile(loss='mean_squared_error', optimizer='adam')
    # model.fit(X_train, y_train, batch_size=20, epochs=10)   # loss: 2.5

    # model.add(Dense(units=6, activation='relu', input_dim=len(X_train.columns)))
    # model.add(Dense(units=12, activation='tanh'))
    # model.add(Dense(units=1))
    # model.compile(loss='mean_squared_error', optimizer='adam')
    # model.fit(X_train, y_train, batch_size=50, epochs=250)  # loss: 0.2

    if run_mode==1:
        model.save('xy_model')  # save the model to a folder?

elif run_mode==3:
    print("\nLoading stored model...")
    model = load_model('xy_model') # load the model from the given folder

## show model performance
test_loss = model.evaluate(X_test, y_test, verbose=0)
print("\nModel performance:")
print(f"> Loss on test set: {test_loss:.3f}")


############################################################
# further testing
############################################################

def rand_input():
    '''input range of the function'''
    return random.random()*10-5

def function_to_predict(x1,x2):
    '''the function to predict'''
    return x2*math.sin(x1) - x1*math.cos(x2)

x_hat = []
y_true = []
for _ in range(5):
    x1, x2 = rand_input(), rand_input()
    x_hat.append([x1,x2])
    y_true.append(function_to_predict(x1,x2))

y_hat = model.predict(np.array(x_hat),verbose=0)

print("\nMore test results using random inputs:")
for i in range(len(y_true)):
    print(f"> x1,x2 = {x_hat[i][0]:+.2f}, {x_hat[i][1]:+.2f}; ",end='')
    print(f"predicted = {y_hat[i][0]:+.2f}; actual={y_true[i]:+.2f}; "
          f"diff={abs(y_true[i]-y_hat[i][0]):.2f}")
