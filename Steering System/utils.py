import cv2

import numpy as np

from scipy.misc import imread, imresize
from sklearn.utils import shuffle

def resize(imgs, shape=(64, 64, 3)):
    height, width, channels = shape
    imgs_resized = np.empty([len(imgs), height, width, channels])
    for i, img in enumerate(imgs):
        imgs_resized[i] = imresize(img, shape)

    return imgs_resized

def rgb2gray(imgs):
    return np.mean(imgs, axis=3, keepdims=True)
    #return imgs
    
def normalize(imgs):
    return imgs / (255.0 / 2.0) - 1

def preprocess(imgs):
    imgs_processed = resize(imgs)
    imgs_processed = rgb2gray(imgs_processed)
    imgs_processed = normalize(imgs_processed)

    return imgs_processed

####################################################

def read_imgs(img_paths):
    imgs = np.empty([len(img_paths), 160, 320, 3])

    for i, path in enumerate(img_paths):
        imgs[i] = imread(path)

    return imgs

def random_flip(imgs, angles):
    new_imgs = np.empty_like(imgs)
    new_angles = np.empty_like(angles)
    for i, (img, angle) in enumerate(zip(imgs, angles)):
        if np.random.choice(2):
            new_imgs[i] = np.fliplr(img)
            new_angles[i] = angle * -1
        else:
            new_imgs[i] = img
            new_angles[i] = angle

    return new_imgs, new_angles

def augment(imgs, angles):
    imgs_augmented, angles_augmented = random_flip(imgs, angles)

    return imgs_augmented, angles_augmented

def gen_batches(imgs, angles, batch_size):
    num_elts = len(imgs)

    while True:
        indeces = np.random.choice(num_elts, batch_size)
        batch_imgs_raw, angles_raw = read_imgs(imgs[indeces]), angles[indeces].astype(float)

        batch_imgs, batch_angles = augment(preprocess(batch_imgs_raw), angles_raw)

        yield batch_imgs, batch_angles