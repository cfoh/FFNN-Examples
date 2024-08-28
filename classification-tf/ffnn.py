############################################################
# install the following dependencies:
# > pip install numpy pandas scikit-learn tensorflow
############################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

############################################################
# the data file carries the data (in csv format) to classify,
# it contains 5 columns:
# - sepal length in cm
# - sepal width in cm
# - petal length in cm
# - petal width in cm
# - class: 
#   - Iris Setosa
#   - Iris Versicolour
#   - Iris Virginica
############################################################

df = pd.read_csv("iris.data", header=None)

X = df.iloc[:, :-1]  # All columns except the last one as features
y = df.iloc[:, -1]   # The last column as the target label

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y) # encode labels to integers

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

    model.add(Dense(units=10, activation='relu', input_dim=len(X_train.columns)))
    model.add(Dense(units=20, activation='relu'))
    model.add(Dense(units=len(label_encoder.classes_), activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=24, epochs=100) 

    if run_mode==1:
        model.save('ffnn_model')  # save the model to a folder

elif run_mode==3:
    print("\nLoading stored model...")
    model = load_model('ffnn_model') # load the model from the given folder

## show model performance

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("\nModel performance:")
print(f"> Test Loss: {test_loss:.3f}")
print(f"> Test Accuracy: {test_accuracy:.3f}")

############################################################
# further testing
############################################################

## pick one to test:
# feature = [6.2,2.9,4.3,1.3]; label = "Iris-versicolor"
# feature = [5.4,3.4,1.7,0.2]; label = "Iris-setosa"
feature = [5.8,2.7,5.1,1.9]; label = "Iris-virginica"

feature_array = np.array([feature])  
predictions = model.predict(feature_array)
predicted_labels = predictions.argmax(axis=1)
predicted_original_labels = label_encoder.inverse_transform(predicted_labels)

print()
print(f"NN_output = {predictions[0]}")
print(f"             {predictions[0][0]:.5f} prob for Iris-Setosa")
print(f"             {predictions[0][1]:.5f} prob for Iris-Versicolour")
print(f"             {predictions[0][2]:.5f} prob for Iris Virginica")
print(f"Prediction   = {predicted_original_labels[0]}")
print(f"Actual_Label = {label}")
