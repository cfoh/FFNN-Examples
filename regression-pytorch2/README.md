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
```

We have 5000 instances in our dataset, 80% (or 4000) are used for training. Since each batch size is 50, the FFNN will be updated 80 times for each epoch, and we run 200 epochs. The following shows the final few epochs. In the final epoch, we achieve a loss of around 0.0215.

```
...
Epoch [194/200], Loss: 0.0220
Epoch [195/200], Loss: 0.0219
Epoch [196/200], Loss: 0.0218
Epoch [197/200], Loss: 0.0217
Epoch [198/200], Loss: 0.0217
Epoch [199/200], Loss: 0.0216
Epoch [200/200], Loss: 0.0215
```

The following is the outcome using the stored model:

```
Loading stored model...

Model performance:
> Loss on test set: 0.014

More test results using random inputs:
> Re(z),Im(z) = -1.43, -0.05; 
  >> Predicted Real = -0.85; actual=-0.82; diff=0.03
  >> Predicted Imag = -1.06; actual=-0.96; diff=0.10
> Re(z),Im(z) = +1.17, +0.07; 
  >> Predicted Real = +3.68; actual=+3.68; diff=0.00
  >> Predicted Imag = -0.77; actual=-0.69; diff=0.08
> Re(z),Im(z) = +1.98, +0.14; 
  >> Predicted Real = +7.63; actual=+7.88; diff=0.26
  >> Predicted Imag = -0.19; actual=-0.18; diff=0.01
> Re(z),Im(z) = +1.40, -0.77; 
  >> Predicted Real = +4.28; actual=+4.17; diff=0.11
  >> Predicted Imag = -4.65; actual=-4.71; diff=0.06
> Re(z),Im(z) = -0.94, +0.43; 
  >> Predicted Real = -0.97; actual=-1.18; diff=0.21
  >> Predicted Imag = -0.97; actual=-0.95; diff=0.02
```

The model is now ready for prediction. Do the following to predict a single point:

```python
z_real, z_imag = -0.8, 1.6
input_tensor = torch.tensor([z_real,z_imag], dtype=torch.float32).to(device)
y = model(input_tensor).cpu()
y_real = y[0].item()
y_imag = y[1].item()
```

![ffnn](https://github.com/cfoh/FFNN-Examples/assets/51439829/3dc94771-da56-4f01-93fc-c4e2ab3e76c0)
