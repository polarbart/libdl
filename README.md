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
 - your python version should be 3.5 or higher
 - the library depends on the following packages:
   - `numpy`
   - `matplotlib`
   - `scikit-image>=0.14`
 - you can install the dependencies with `pip install -r requirements.txt`
 
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

If you have any questions just write me an email (jhansjakob@googlemail.com or julius.hansjakob@tum.de). I'll try to answer as soon as possible.

## Top Level Documentation
The top level documentation can be found [here](https://gitlab.lrz.de/hansjakob/libdl/wikis/Top-Level-Documentation).

## Building the Tests
 - `mkdir build`
 - `cd build`
 - `cmake ..`
 - `make libdl_tests`
 - `make test`