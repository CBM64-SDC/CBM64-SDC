# CBM64-SDC

## Steering Control by a Convolutional Neural Network

### Current Model Configurations:
#### Inspired by Nvidia End-to-End Learning for Self Driving Car

Validation/Train data splitting rate = 0.7

batch size = 40

epochs = 10

Preprocessing: Brightness Augmentation, Resizing and Cropping, RGB to YUV Conversion, and Normalization.

Input shape after pre-processing = 66, 200, 3

Learning Rate: 1e-4

Throttle (drive.py): Controlled by PID controller.
