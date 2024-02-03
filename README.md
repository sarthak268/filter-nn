# Sobel Filter using Neural Networks

This repository contains the code for the implementation of a sobel filter using neural networks.


## Index

1. [Environment Setup](#setup)
2. [Training](#training)
3. [Testing](#testing)

## Environment Setup

In order to build a ```conda``` environment for running our model, run the following command:
```
conda env create -f environment.yml
```

Activate environment using:
```
conda activate sobel
```

## Training 

To train our model, run the following command:
```
CUDA_VISIBLE_DEVICES=ID python train.py
```

You can choose to modify the hyperparameters for training in ```config.json```, including changing the filter you would want to us. You can also add your own custom filter that you want the model to learn in the ```get_gt_values``` function in the ```dataset.py``` file.

## Testing 

To evaluate our model on your test image, place the image in root directory with the name ```test_image.jpg```
The, run the following command:

```
CUDA_VISIBLE_DEVICES=ID python test.py
```