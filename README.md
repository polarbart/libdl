# My super awesome deep learning library

## Installation
1. Compile the library
 - `mkdir build`
 - `cd build`
 - `cmake ..`
 - `make libdl_python`
2. Then copy the compiled library to the pylibdl folder
 - The compiled library is located at `build/python` and it's filename starts with `libdl_python`
 - If you are still in the build folder you can copy it with`cp python/libdl_python.* ../pylibdl`
3. python requirements  
 - Your python version should be 3.5 or higher
 - The library depends on the following packages:
   - `numpy`
   - `matplotlib`
   - `scikit-image>=0.14`
 - You can install the dependencies with `pip install -r requirements.txt`
 
## XOR Problem
See [XOR.ipynb](XOR.ipynb)

## MNIST
See [MNIST.ipynb](MNIST.ipynb)  
Training and testing takes about 30 seconds. 

## Final Project
First download the [distracted driver dataset](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data) and extract it into the folder `distracted_driver`.  
Don't forget to extract the images from `ìmgs.zip`.  
In the end your folder structure should look like this:
```
|-- libdl/  
|   |-- distracted_driver/  
|   |   |-- test/  
|   |   |-- train/  
|   |   |-- driver_imgs_list.csv  
|   |   |-- sample_submission.csv  
|   |-- ...  
```
For my final project see [Final.ipynb](Final.ipynb). Some code fragments are also located at [utils.py](utils.py).  
(Sadly the gitlab ipynb viewer does not render everything correctly i.e. better view it on your local jupyter server)

## Top Level Documentation
The top level documentation can be found [here](https://gitlab.lrz.de/hansjakob/libdl/wikis/Top-Level-Documentation).

## Building the Tests
 - `mkdir build`
 - `cd build`
 - `cmake ..`
 - `make libdl_tests`
 - `make test`

# Examples
My library uses a dynamic computational graph and is closely modeled after PyTorch. 
I.e. you don't have to define your computational graph in beforehand. It is created implicitly while you do your compuations.

This is achieved with a slightly more advanced tensor class, which not only holds a reference to its data `Tensor::data`, but also a reference to its gradient `Tensor::data` and a reference to the operation that created that tensor `Tensor::gradFn`.

When the `Tensor::backward()` function is called, the gradients for all predecessors are computed. I.e. the gradient of the tensor calling 'backward' with respect to each predecessor.

The tensor class can be found at `src/Tensor.h`. The operations like add, conv, matmul, batchnorm, ... can be found at `src/ops/`. Each operation has a static method which computes the 'forward pass' and a method called 'computeGradients' which computes the 'backward pass', i.e. the gradients for its parents.

## A Simple Example
Let's look at the function
```math
f(a, b) = a^2 + b^3
```
The derivatives towards $`a`$ and $`b`$ are: 
```math 
\frac{\partial f}{\partial a} = 2a  
```

```math 
\frac{\partial f}{\partial b}  = 3b^2
```
If we evaluate $`f`$ at $`f(-3, 4)`$  
```math 
f(2, 3) = (-3)^2 + 4^3 = 9 + 64 = 73
```
The derivatives with respect to $`a`$ and $`b`$ are:  
```math 
\frac{\partial f}{\partial a} = 2\cdot (-3) = -6  
```

```math 
\frac{\partial f}{\partial b}  = 3\cdot 4^2 = 48
```

Let's to these calculations in python:

```python
>>> from pylibdl import tensor
>>> import numpy as np
>>> a = tensor(np.array([-3]), requires_grad=True)  # gradFn = Leaf
>>> b = tensor(np.array([4]), requires_grad=True)  # gradFn = Leaf
>>> y = a**2 + b**3  # gradFn of y is Add<float, 1, 1>
>>> print(y.data)
[ 73 ]
>>> print(a.grad, b.grad)
None None
>>> y.backward()  # computes the gradient for all predecessors
>>> print(a.grad, b.grad)
[ -6 ] [ 48 ]
```

The animation below visualizes what happens under the hood. 
![animation](animation.gif)

## XOR Example
 - The example below shows how this library can be used to train neural networks
 - You can look at `pylibdl/modules.py` if you want to get a deeper understanding of how the layers/modules are implemented

```python
import numpy as np
import pylibdl as libdl
from pylibdl.modules import Module, Sequential, Linear, Sigmoid
from pylibdl.optim import Adam

# hyperparamter
hidden_units = 2
lr = 1e-2
epochs = 10000
log_every = 1000

# dataset
X = libdl.tensor(np.array([[0, 0, 1, 1], [0, 1, 0, 1]]))
y = libdl.tensor(np.array([[0, 1, 1, 0]]))

# this is the neural network
class XORClassifier(Module):
    def __init__(self):
        super().__init__()
        # one hidden layer with 'hidden_units' hidden neurons
        # after every layer we apply a sigmoid activation function
        # the first layer has two input neurons and the last layer has one output neuron
        # Sequential applies the layers in a sequential order
        self.l1 = Sequential(Linear(2, hidden_units), Sigmoid()) 
        self.l2 = Sequential(Linear(hidden_units, 1), Sigmoid())

    def forward(self, x):
        # this method does the forward pass
        # i.e. it runs 'x' through the network and returns the neworks output
        h1 = self.l1(x)
        o = self.l2(h1)
        return o

# instanciate the model
model = XORClassifier()

# use adam optimzer
optimizer = Adam(model.parameter(), lr)

print("epoch |  0^0 |  0^1 |  1^0 |  1^1 | loss")
for epoch in range(epochs):
    
    # forward pass, behind the scenes XORClassifier.forward is called
    pred = model(X)
    
    # compute the loss between the predictions and the true labels
    loss = libdl.mean((pred - y)**2)
    
    # backpropagate the loss
    loss.backward()
    
    # change the parameters of the model so that the loss is minimized and then reset the gradients
    optimizer.step()
    optimizer.zero_grad()
    
    if (epoch % log_every) == 0 or epoch == (epochs - 1):
        print(f"{epoch:5d} | {pred.data[0, 0]:.2f} | {pred.data[0, 1]:.2f} | {pred.data[0, 2]:.2f} | {pred.data[0, 3]:.2f} | {loss.data[0]:.6f}")

epoch |  0^0 |  0^1 |  1^0 |  1^1 | loss
    0 | 0.62 | 0.57 | 0.60 | 0.56 | 0.260224
 1000 | 0.12 | 0.86 | 0.90 | 0.11 | 0.013432
 2000 | 0.06 | 0.93 | 0.95 | 0.06 | 0.003885
 3000 | 0.04 | 0.95 | 0.96 | 0.04 | 0.001742
 4000 | 0.03 | 0.97 | 0.97 | 0.03 | 0.000904
 5000 | 0.02 | 0.97 | 0.98 | 0.02 | 0.000502
 6000 | 0.02 | 0.98 | 0.99 | 0.02 | 0.000289
 7000 | 0.01 | 0.98 | 0.99 | 0.01 | 0.000170
 8000 | 0.01 | 0.99 | 0.99 | 0.01 | 0.000102
 9000 | 0.01 | 0.99 | 0.99 | 0.01 | 0.000061
 9999 | 0.01 | 0.99 | 0.99 | 0.01 | 0.000037
```
