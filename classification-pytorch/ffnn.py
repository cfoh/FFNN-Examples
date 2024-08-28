############################################################
# install the following dependencies:
# > pip install numpy pandas scikit-learn torch
############################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim

## check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

## load and preprocess the data
df = pd.read_csv("iris.data", header=None)

X = df.iloc[:, :-1].values  # all columns except the last one as features
y = df.iloc[:, -1].values   # the last column as the target label

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # encode labels to integers

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

## define the model
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

############################################################
# training
############################################################

run_mode = 3  # 1: train & save the model into a file
              # 2: train but don't save the model
              # 3: load the saved model

if run_mode == 1 or run_mode == 2:
    model = FFNN(input_dim=X_train.shape[1], 
                 output_dim=len(label_encoder.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    batch_size = 24

    for epoch in range(num_epochs):

        permutation = torch.randperm(X_train.size()[0])
        epoch_loss = 0.0
        correct = 0
        total = 0

        for i in range(0, X_train.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            ## accumulate loss
            epoch_loss += loss.item()

            ## calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        ## calculate average loss and accuracy for the epoch
        epoch_loss /= len(X_train) // batch_size
        epoch_accuracy = correct / total

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    if run_mode == 1:
        torch.save(model.state_dict(), 'ffnn_model.pth')

elif run_mode == 3:
    print("\nLoading stored model...")
    model = FFNN(input_dim=X_test.shape[1], 
                 output_dim=len(label_encoder.classes_)).to(device)
    model.load_state_dict(torch.load('ffnn_model.pth'))
    model.eval()

## evaluate model performance
with torch.no_grad():
    outputs = model(X_test)
    loss = nn.CrossEntropyLoss()(outputs, y_test).item()
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)

print("\nModel performance:")
print(f"> Test Loss: {loss:.3f}")
print(f"> Test Accuracy: {accuracy:.3f}")

# Further testing
# feature = [6.2,2.9,4.3,1.3]; label = "Iris-versicolor"
# feature = [5.4,3.4,1.7,0.2]; label = "Iris-setosa"
feature = [5.8,2.7,5.1,1.9]; label = "Iris-virginica"

feature_tensor = torch.tensor([feature], dtype=torch.float32)
predictions = model(feature_tensor)
predicted_labels = torch.argmax(predictions, dim=1)
predicted_original_labels = label_encoder.inverse_transform(predicted_labels.numpy())

print()
print(f"NN_output = {predictions[0].detach().numpy()}")
print(f"             {predictions[0][0]:.5f} prob for Iris-Setosa")
print(f"             {predictions[0][1]:.5f} prob for Iris-Versicolour")
print(f"             {predictions[0][2]:.5f} prob for Iris Virginica")
print(f"Prediction   = {predicted_original_labels[0]}")
print(f"Actual_Label = {label}")
