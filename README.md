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