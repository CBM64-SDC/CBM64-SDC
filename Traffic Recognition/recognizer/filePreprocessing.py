import numpy as np
from skimage import io, color, exposure, transform
import pandas as pd
import os
import glob
import h5py

NUM_CLASSES = 43
IMG_SIZE = 48

def preprocess_img(img):
    # Histogram normalization in y
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img,-1)

    return img

def get_class(img_path):
    return int(img_path.split('/')[-2])

if __name__ == '__main__':
    trainFound = 0
    testFound = 0
    files = os.listdir('../data/')
    for f in files:
        if f == 'train.h5':
            trainFound = 1
        if f == 'test.h5':
            testFound = 1
        if trainFound and testFound:
            break
    
    if trainFound == 0:
        print("Processing Training Files...")
        root_dir = '../data/GTSRB/Final_Training/Images/'
        imgs = []
        labels = []

        all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
        np.random.shuffle(all_img_paths)

        files = os.listdir('../data/')
        for img_path in all_img_paths:
            try:
                img = preprocess_img(io.imread(img_path))
                label = get_class(img_path)
                imgs.append(img)
                labels.append(label)

                if len(imgs)%1000 == 0: print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
            except (IOError, OSError):
                print('missed', img_path)
                pass

        X = np.array(imgs, dtype='float32')
        Y = np.array(labels, dtype='uint8')
        imgs = labels = None

        with h5py.File('../data/train.h5','w') as hf:
            hf.create_dataset('imgs', data=X)
            hf.create_dataset('labels', data=Y+1)

    if testFound == 0:
        print("Processing Test Files...")
        test = pd.read_csv('../data/GTSRB/GT-final_test.csv',sep=';')

        X_test = []
        y_test = []
        i = 0
        for file_name, class_id  in zip(list(test['Filename']), list(test['ClassId'])):
            img_path = os.path.join('../data/GTSRB/Final_Test/Images/',file_name)
            image = preprocess_img(io.imread(img_path))
            X_test.append(image)
            y_test.append(class_id+1)
            
        X_test = np.array(X_test, dtype='float32')
        y_test = np.array(y_test, dtype='uint8')

        with h5py.File('../data/test.h5','w') as hf:
            hf.create_dataset('imgs', data=X_test)
            hf.create_dataset('labels', data=y_test)