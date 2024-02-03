# Sobel Filter using Neural Networks

This repository contains the code for the implementation of a sobel filter using neural networks.

## Index

1. [Understanding the Code](#understanding)
2. [Environment Setup](#setup)
3. [Training](#training)
4. [Testing](#testing)
5. [Results](#results)

## Understanding the Code

```python
function train(input):
    // Step 1: Initialize loss functions, train and val dataset, network  

    // Step 2: Loop through the dataset
    for batch in input:
        // Step 3: Compute model output
        // Step 4: Compute loss between output and GT
        // Step 5: Backprop and update weights

    // Step 6: Run Steps 2-5 for validation dataset and log the loss
    return model
```

```python
function generateData(input):
    // Step 1: Load COCO dataset
    // Step 2: Compute Sobel filter on each sample 
    // Step 3: Wrap this dataset around a dataloader 

    return dataloader
```

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

You can choose to modify the hyperparameters for training in ```config.json```. 

In the same file, you can modify the filter you would want to us (variable for that is called ```filter```). The possible values it can take are ```sobel```, ```sym_sobel```, ```avg```, ```blur```, and ```custom```.
You can also add your own custom filter that you want the model to learn in the ```get_gt_values``` function in the ```dataset.py``` file.

## Testing 

To evaluate our model on your test image, place the image in root directory with the name ```test_image.jpg```
The, run the following command:

```
CUDA_VISIBLE_DEVICES=ID python test.py
```

## Results

Here are some of the finding from the evaluation of the trained models.

![Sobel](media/sobel_layers.png)

For the sobel filter, 3 convolutional layers seem to be optimal. Performance with 2 and 4 convolutional layers also saturate to a similar loss (both train and validation) but with 3 layers, the convergence is faster. Sobel filter does not work well with just one layer (loss converges to 0.3 which is significantly higher than that of >1 layers).

<p align="center">
  <img src="media/sobel/images_input.png" width="30%" alt="Input Images">
  <img src="media/sobel/images_target.png" width="30%" alt="Target Images">
  <img src="media/sobel/images_output.png" width="30%" alt="Output Images">
</p>

![Blur](media/blur_layers.png)

For blurring filter, again 3 convolutional layers lead to faster convergence but eventually, the one layer network reaches a slightly lower loss (both train and validation). 

<p align="center">
  <img src="media/sym_sobel/images_input.png" width="30%" alt="Input Images">
  <img src="media/sym_sobel/images_target.png" width="30%" alt="Target Images">
  <img src="media/sym_sobel/images_output.png" width="30%" alt="Output Images">
</p>

I also tested out a sobel filter that uses wrap boundaries to reduce artifacts at the boundaries of the image. The loss for the model with wrap boundaries is a little higher because the convolutional layers inherently use zero value paddings. This one works well for any number of layers >= 3.