############################################################
# install the following dependencies:
# > pip install numpy pandas scikit-learn torch
############################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

## check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

############################################################
# the csv specifies the complex function to predict,
# it contains 4 columns:
# - column 1: `Real(z)`, float between -2.0 & 2.0
# - column 2: `Imag(z)`, float between -2.0 & 2.0
# - column 3: `Real(f(z))`,  float Re(z**2 + 2*z - 1j)
# - column 4: `Imag(f(z))`,  float Im(z**2 + 2*z - 1j)
############################################################

## load data
df = pd.read_csv("sample_data.csv")
X = df[["Real(z)", "Imag(z)"]].values          # the input columns
y = df[["Real(f(z))", "Imag(f(z))"]].values    # the output columns

## split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

## define the model
class FFNN(nn.Module):
    def __init__(self):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(2, 6)
        self.fc2 = nn.Linear(6, 12)
        self.fc3 = nn.Linear(12, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

model = FFNN().to(device)

############################################################
# training
############################################################

run_mode = 3  # 1: train & save the model into a file
              # 2: train but don't save the model
              # 3: load the saved model

if run_mode == 1 or run_mode == 2:
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    ## training loop
    num_epochs = 200
    batch_size = 50

    for epoch in range(num_epochs):
        ## mini-batch gradient descent
        permutation = torch.randperm(X_train.size()[0])
        running_loss = 0.0
        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/200], Loss: {running_loss / batch_size:.4f}")

    if run_mode == 1:
        torch.save(model.state_dict(), 'z_model.pth')

elif run_mode == 3:
    print("\nLoading stored model...")
    model = FFNN()
    model.load_state_dict(torch.load('z_model.pth'))
    model.eval()

## evaluate model performance
with torch.no_grad():
    y_pred = model(X_test)
    test_loss = nn.MSELoss()(y_pred, y_test).item()

print("\nModel performance:")
print(f"> Loss on test set: {test_loss:.3f}")

############################################################
# further testing
############################################################
def f(z):
    return z**2 + 2*z - 1j

num_samples = 5   # number of samples to test

## generate random complex numbers within a specified range
z_real = np.random.uniform(-2, 2, num_samples)
z_imag = np.random.uniform(-2, 2, num_samples)
z_samples = z_real + 1j * z_imag
f_samples = f(z_samples)  # output from f(z)
f_real, f_imag = f_samples.real, f_samples.imag

## predict the outputs
z_hat = torch.tensor(np.dstack((z_real, z_imag))[0], dtype=torch.float32)
f_hat = model(z_hat).detach().numpy()

print("\nMore test results using random inputs:")
for i in range(num_samples):
    print(f"> Re(z),Im(z) = {z_hat[i][0]:+.2f}, {z_hat[i][1]:+.2f}; ")
    print(f"  >> Predicted Real = {f_hat[i][0]:+.2f}; actual={f_real[i]:+.2f}; "
          f"diff={abs(f_real[i]-f_hat[i][0]):.2f}")
    print(f"  >> Predicted Imag = {f_hat[i][1]:+.2f}; actual={f_imag[i]:+.2f}; "
          f"diff={abs(f_imag[i]-f_hat[i][1]):.2f}")
