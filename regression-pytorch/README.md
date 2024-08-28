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

The following is the default FFNN design in the code using pytorch:

```python
    class FFNN(nn.Module):
        def __init__(self):
            super(FFNN, self).__init__()
            self.fc1 = nn.Linear(2, 6)
            self.fc2 = nn.Linear(6, 12)
            self.fc3 = nn.Linear(12, 1)
            self.relu = nn.ReLU()
            self.tanh = nn.Tanh()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.tanh(self.fc2(x))
            x = self.fc3(x)
            return x
```

We have 5000 instances in our dataset, 80% (or 4000) are used for training. Since each batch size is 50, the FFNN will be updated 80 times for each epoch, and we run 200 epochs. The following shows the final few epochs. In the final epoch, we achieve a loss of below 0.3.

```
...
Epoch [196/200], Loss: 0.3367
Epoch [197/200], Loss: 0.3000
Epoch [198/200], Loss: 0.3540
Epoch [199/200], Loss: 0.3903
Epoch [200/200], Loss: 0.2703
```

The following is the outcome using the stored model:

```
Loading stored model...

Model performance:
> Loss on test set: 0.292

More test results using random inputs:
> x1,x2 = +2.06, +2.22; predicted = +2.90; actual=+3.21; diff=0.30
> x1,x2 = -0.76, +3.74; predicted = -3.67; actual=-3.20; diff=0.47
> x1,x2 = -1.66, -0.11; predicted = +1.52; actual=+1.77; diff=0.25
> x1,x2 = +4.72, +3.14; predicted = +0.91; actual=+1.57; diff=0.67
> x1,x2 = +2.33, -3.64; predicted = +0.02; actual=-0.59; diff=0.61
```

The model is now ready for further prediction. Do the following to predict a single point:

```python
x1, x2 = -0.8, 3.6
input_tensor = torch.tensor([[x1,x2]], dtype=torch.float32).to(device)
y = model(input_tensor).cpu().item()
```

![ffnn](https://github.com/cfoh/FFNN-Examples/assets/51439829/838c3a4a-7951-4bb9-a187-c2ba3d69fb62)
