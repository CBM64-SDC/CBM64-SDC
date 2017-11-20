# CBM64-SDC

## Current Model Configurations:
Inspired by Nvidia End-to-End Learning for Self Driving Car

Number of Parameters: >= 5,000,000

Number of epochs: 10

Input Shape: 64 x 64

Preprocessing: Brightness Augmentation, Resizing and Cropping, Grayscale Conversion, and Normalization.

Learning Rate: 1e-4

Throttle (drive.py): Controlled by PID controller.

################################### Datasets ############################################

* Haar Cascade datasets for various objects: https://github.com/opencv/opencv/tree/master/data/haarcascades

* 38K img dataset for track1 - Udacity self driving car simulator: http://www.mediafire.com/file/5icq78dcz9ac1d8/38KImgDataset.zip

#########################################################################################
