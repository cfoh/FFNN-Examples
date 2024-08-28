############################################################
# install the following dependencies:
# > pip install numpy pandas scikit-learn tensorflow
############################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

############################################################
# the csv specifies the complex function to predict,
# it contains 4 columns:
# - column 1: `Real(z)`, float between -2.0 & 2.0
# - column 2: `Imag(z)`, float between -2.0 & 2.0
# - column 3: `Real(f(z))`,  float Re(z**2 + 2*z - 1j)
# - column 4: `Imag(f(z))`,  float Im(z**2 + 2*z - 1j)
############################################################

df = pd.read_csv("sample_data.csv")
X = df[["Real(z)","Imag(z)"]]          # the input columns
y = df[["Real(f(z))","Imag(f(z))"]]    # the output column

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
    model.add(Dense(units=len(y_train.columns)))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(X_train, y_train, batch_size=50, epochs=200)  # loss: 0.17

    if run_mode==1:
        model.save('z_model')  # save the model to a folder?

elif run_mode==3:
    print("\nLoading stored model...")
    model = load_model('z_model') # load the model from the given folder

## show model performance
test_loss = model.evaluate(X_test, y_test, verbose=0)
print("\nModel performance:")
print(f"> Loss on test set: {test_loss:.3f}")


############################################################
# further testing
############################################################

def f(z):
    return z**2 + 2*z - 1j

num_samples = 5   # number of samples to test

## Generate random complex numbers within a specified range
z_real = np.random.uniform(-2, 2, num_samples)
z_imag = np.random.uniform(-2, 2, num_samples)
z_samples = z_real + 1j * z_imag
f_samples = f(z_samples) # output from f(z)
f_real, f_imag = f_samples.real, f_samples.imag

## predict the outputs
z_hat = np.dstack((z_real,z_imag))[0]
f_hat = model.predict(z_hat,verbose=0)

print("\nMore test results using random inputs:")
for i in range(num_samples):
    print(f"> Re(z),Im(z) = {z_hat[i][0]:+.2f}, {z_hat[i][1]:+.2f}; ")
    print(f"  >> Predicted Real = {f_hat[i][0]:+.2f}; actual={f_real[i]:+.2f}; "
          f"diff={abs(f_real[i]-f_hat[i][0]):.2f}")
    print(f"  >> Predicted Imag = {f_hat[i][1]:+.2f}; actual={f_imag[i]:+.2f}; "
          f"diff={abs(f_imag[i]-f_hat[i][1]):.2f}")
