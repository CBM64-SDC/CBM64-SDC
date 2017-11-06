import cv2
import csv

import numpy as np
import matplotlib.pyplot as plt

from scipy.misc import imread, imresize
from sklearn.utils import shuffle

from os import listdir
from os.path import join

from keras.preprocessing.image import img_to_array, load_img

def fixPath(base_d, path, append=False):
    base_dir = base_d
    img_files = listdir(path+'/IMG/')

    x = 0
    mode = 'w'
    if append:
        mode = 'a'
    with open('../../data/driving_log_clean.csv', mode) as wf:
        with open(path+'/driving_log.csv', 'r') as rf:
            reader = csv.reader(rf)
            writer = csv.writer(wf)
            if x == 0 and mode == 'w':
                wf.write('center,left,right,steering\n')
                x = 1
            for row in reader:
                rel_path = row[0].split('/')[-1]
                rel_path2 = row[1].split('/')[-1]
                rel_path3 = row[2].split('/')[-1]
                real_path = join(base_dir, rel_path)
                real_path2 = join(base_dir, rel_path2)
                real_path3 = join(base_dir, rel_path3)
                s = ""
                if rel_path in img_files:
                    s += real_path
                if rel_path2 in img_files:
                    s += ','
                    s += real_path2
                if rel_path3 in img_files:
                    s += ','
                    s += real_path3
                s += ','
                s += row[3]
                wf.write(s+'\n')

def visualize_distribution(data):
    num_bins = 23
    avg_samples_per_bin = len(data)/num_bins
    hist, bins = np.histogram(data, num_bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.plot((np.min(data), np.max(data)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
    plt.show()

################# Preprocessing ####################

def resize(img, shape=(64, 64, 3)):
    cropped_image = img[55:135, :, :]
    processed_image = imresize(cropped_image, shape)
    return processed_image

def rgb2gray(img):
    ret = np.mean(img, axis=2, keepdims=True)
    return ret
    
def normalize(img):
    img = (img / 255) - 0.5
    return img

def preprocess(img):
    img1 = resize(img)
    img2 = rgb2gray(img1)
    img3 = normalize(img2)
    
    return img3

def augment_brightness(img):
    newimg = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)

    # randomly generate the brightness reduction factor
    # Add a constant so that it prevents the image from being completely dark
    random_bright = .30+np.random.uniform()

    # Apply the brightness reduction to the V channel
    newimg[:,:,2] = newimg[:,:,2]*random_bright

    # convert to RBG again
    newimg = cv2.cvtColor(newimg,cv2.COLOR_HSV2RGB)
    return newimg

####################################################

################# PID Controller ###################

class PID:
    def __init__(self):
        self.preverror = 0
        self.integral = 0

    def calc(self, speed, mx, dt = 0.1, kp = 0.1, kd = 0.01, ki = 0.5):

        # Calculate the error
        error = mx - speed

        # Proportional Term
        Pout = kp * error

        # Integral Term
        self.integral += (error * dt)
        Iout = ki * self.integral

        # Derivative Term
        derivative = (error - self.preverror) / dt
        Dout = kd * derivative

        throttle = Pout + Iout + Dout

        self.preverror = error

        if throttle > 1:
            throttle = 1
        elif throttle < 0:
            throttle = 0

        return throttle

####################################################

def read(row):
    steering = row['steering']

    camera = np.random.choice(['center', 'right', 'left'])

    if camera == 'Left':
        steering += 0.25
    elif camera == 'Right':
        steering -= 0.25

    img = load_img(row[camera])
    img = img_to_array(img)

    coin = np.random.random()

    if coin > 0.5:
        steering = -1*steering
        img = cv2.flip(img, 1)

    img = augment_brightness(img)
    img = preprocess(img)

    return img, steering

def gen_batches(data, batch_size=32):
    batches_per_epoch = data.shape[0] // batch_size

    i = 0
    while (True):
        s = i * batch_size
        e = s + batch_size - 1

        X_batch = np.zeros((batch_size, 64, 64, 1), dtype=np.float32)
        y_batch = np.zeros((batch_size,), dtype=np.float32)

        j = 0

        for index, row in data.loc[s:e].iterrows():
            X_batch[j], y_batch[j] = read(row)
            j += 1

        i += 1
        if i == batches_per_epoch - 1:
            i = 0
        
        yield X_batch, y_batch