# Creating a CIFAR 10 Convolution Neural Network

    This repo contains a CNN for training CIFAR 10 Dataset.

`model.py` file contains the CNN Model. It has `FinalModel` which is the final lighter model with under 50k parameters and produces over 70% train and test accuracy for the CIFAR 10 Dataset.

Model Summary is as Follows -

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
              ReLU-2           [-1, 32, 32, 32]               0
       BatchNorm2d-3           [-1, 32, 32, 32]              64
           Dropout-4           [-1, 32, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]           9,216
              ReLU-6           [-1, 32, 32, 32]               0
       BatchNorm2d-7           [-1, 32, 32, 32]              64
           Dropout-8           [-1, 32, 32, 32]               0
            Conv2d-9           [-1, 16, 32, 32]             512
        MaxPool2d-10           [-1, 16, 16, 16]               0
           Conv2d-11           [-1, 16, 16, 16]           2,304
             ReLU-12           [-1, 16, 16, 16]               0
      BatchNorm2d-13           [-1, 16, 16, 16]              32
          Dropout-14           [-1, 16, 16, 16]               0
           Conv2d-15           [-1, 16, 16, 16]           2,304
             ReLU-16           [-1, 16, 16, 16]               0
      BatchNorm2d-17           [-1, 16, 16, 16]              32
          Dropout-18           [-1, 16, 16, 16]               0
           Conv2d-19           [-1, 16, 16, 16]           2,304
             ReLU-20           [-1, 16, 16, 16]               0
      BatchNorm2d-21           [-1, 16, 16, 16]              32
          Dropout-22           [-1, 16, 16, 16]               0
           Conv2d-23           [-1, 16, 16, 16]             256
        MaxPool2d-24             [-1, 16, 8, 8]               0
           Conv2d-25             [-1, 32, 8, 8]           4,608
             ReLU-26             [-1, 32, 8, 8]               0
      BatchNorm2d-27             [-1, 32, 8, 8]              64
          Dropout-28             [-1, 32, 8, 8]               0
           Conv2d-29             [-1, 32, 8, 8]           9,216
             ReLU-30             [-1, 32, 8, 8]               0
      BatchNorm2d-31             [-1, 32, 8, 8]              64
          Dropout-32             [-1, 32, 8, 8]               0
           Conv2d-33             [-1, 32, 8, 8]           9,216
             ReLU-34             [-1, 32, 8, 8]               0
      BatchNorm2d-35             [-1, 32, 8, 8]              64
          Dropout-36             [-1, 32, 8, 8]               0
        AvgPool2d-37             [-1, 32, 1, 1]               0
           Conv2d-38             [-1, 10, 1, 1]             320
================================================================
Total params: 41,536
Trainable params: 41,536
Non-trainable params: 0
----------------------------------------------------------------
```

Model Accuracy:

- Using Batch Normalisation:
    - Training Accuracy: 79.36
    - Test Accuracy: 78.35

- Using Group Normalisation:
    - Training Accuracy: 78.0
    - Test Accuracy: 76.73

- Using Layer Normalisation:
    - Training Accuracy: 76.76
    - Test Accuracy: 75.87

Observations:
    - Batch Normalisation performed well compared to the Group and Layer Normalisation.
    - Batch Normalisation is less overfit model compared to the Group and Layer Normalisation
    - Model is fully convolution based model without any Fully Connected layers so Group and Layer may not be able to fully take advantage

Incorrect Predictions Images:

![Batch Normalisation Incorrect Predictions](<CleanShot 2023-06-23 at 10.25.37@2x.png>)

![Group Normalisation Incorrect Predictions](<CleanShot 2023-06-23 at 10.26.23@2x.png>)

![Layer Normalisation Incorrect Predictions](<CleanShot 2023-06-23 at 10.26.49@2x.png>)