# My super awesome deep learning library

## Installation
1. Compile the library
 - `mkdir -p build`
 - `cd build`
 - `cmake ..`
 - `make libdl_python`
2. Then copy the compiled library to the pylibdl folder
 - the compiled library is located at `build/python` and it's filename starts with `libdl_python`
 - if you are still in the build folder you can copy it with`cp python/libdl_python.* ../pylibdl`

## XOR Problem
See [XOR.ipynb](XOR.ipynb)

## MNIST
See [MNIST.ipynb](MNIST.ipynb)  
Training and testing takes about 30 seconds. 

## Final Project
See [Final.ipynb](Final.ipynb)
(sadly the gitlab ipynb viewer does not render everything correctly i.e. better view it on your local jupyter server)