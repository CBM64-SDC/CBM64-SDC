import cv2
import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy.misc import imread, imresize
from sklearn.utils import shuffle

from os import listdir
from os.path import join

from keras.preprocessing.image import img_to_array, load_img

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

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

def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :] # remove the sky and the car front

def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def preprocess(image):
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image

def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def random_flip(image, steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle

def random_translate(image, steering_angle, range_x, range_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle

def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

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

def read(center, left, right, steering, range_x=100, range_y=10):
    camera = np.random.choice(3)
    path = ""
    if camera == 0:
        path = center
    elif camera == 1:
        path = left
        steering += 0.20
    else:
        path = right
        steering -= 0.20

    img = mpimg.imread(path)
    img = preprocess(img)
    img, steering = random_flip(img, steering)
    img, steering = random_translate(img, steering, range_x, range_y)
    img = random_shadow(img)
    img = random_brightness(img)

    return img, steering

def gen_batches(X, y, is_training, batch_size=32):
    X_batch = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    y_batch = np.empty(batch_size)
    while True:
        j = 0
        for i in np.random.permutation(X.shape[0]):
            center, left, right = X[i]
            steering = y[i]
            if is_training and np.random.rand() < 0.6:
                img, steering = read(center, left, right, steering)
            else:
                img = preprocess(imread(center))
            X_batch[j] = img
            y_batch[j] = steering
            j += 1
            if j == batch_size:
                break
        yield X_batch, y_batch