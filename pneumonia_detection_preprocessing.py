import os
import fnmatch
import numpy as np
import glob
import random
from keras.preprocessing.image import img_to_array, load_img

#========================================
# CREATE DIRECTORIES AND GET IMAGE COUNTS
#========================================

base_dir = os.path.join(DATA_DIR, 'chest_xray')
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# directory names
normal_train_dir = os.path.join(train_dir, 'NORMAL')
pneumonia_train_dir = os.path.join(train_dir, 'PNEUMONIA')
normal_val_dir = os.path.join(val_dir, 'NORMAL')
pneumonia_val_dir = os.path.join(val_dir, 'PNEUMONIA')
normal_test_dir = os.path.join(test_dir, 'NORMAL')
pneumonia_test_dir = os.path.join(test_dir, 'PNEUMONIA')

# number of .jpeg files in each image category
normal_train_len = len(fnmatch.filter(os.listdir(normal_train_dir), '*.jpeg'))
pneumonia_train_len = len(fnmatch.filter(os.listdir(pneumonia_train_dir), '*.jpeg'))
normal_val_len = len(fnmatch.filter(os.listdir(normal_val_dir), '*.jpeg'))
pneumonia_val_len = len(fnmatch.filter(os.listdir(pneumonia_val_dir), '*.jpeg'))
normal_test_len = len(fnmatch.filter(os.listdir(normal_test_dir), '*.jpeg'))
pneumonia_test_len = len(fnmatch.filter(os.listdir(pneumonia_test_dir), '*.jpeg'))

#===============================
# CONVERT IMAGES TO NUMPY ARRAYS
#===============================

def shuffle(a, b, random_state=25):
    # shuffle two arrays in same order
    assert len(a) == len(b)
    random.seed(random_state)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def create_train_data(normal_train_len, pneumonia_train_len):
    # get number of images for both categories
    train_len = normal_train_len + pneumonia_train_len

    # create generator for image files
    normal_train_images = glob.glob(normal_train_dir + '/*.jpeg')
    pneumonia_train_images = glob.glob(pneumonia_train_dir + '/*.jpeg')

    # create blank arrays for containing data and labels
    train_data = np.zeros((train_len, 256, 256, 1), dtype=np.float32)
    train_labels = np.zeros((train_len, 1), dtype=np.int64)

    # normal cases
    for i, img in enumerate(normal_train_images):
        # read image and resize
        img = img_to_array(load_img(str(img),
                                    target_size=(256, 256),
                                    color_mode='grayscale')) / 255
        train_data[i] = img
        train_labels[i] = 0
        
    # pneumonia cases
    for i, img in enumerate(pneumonia_train_images):
        img = img_to_array(load_img(str(img),
                                    target_size=(256, 256),
                                    color_mode='grayscale')) / 255
        train_data[i + normal_train_len] = img
        train_labels[i + normal_train_len] = 1
        
    # shuffle data
    return shuffle(train_data, train_labels, random_state=14)

train_data, train_labels = create_train_data(normal_train_len, pneumonia_train_len)

def create_val_data(normal_val_len, pneumonia_val_len):
    # get number of images
    val_len = normal_val_len + pneumonia_val_len

    # create generators
    normal_val_images = glob.glob(normal_val_dir + '/*.jpeg')
    pneumonia_val_images = glob.glob(pneumonia_val_dir + '/*.jpeg')

    # create blank arrays
    val_data = np.zeros((val_len, 256, 256, 1), dtype=np.float32)
    val_labels = np.zeros((val_len, 1), dtype=np.int64)

    # normal cases
    for i, img in enumerate(normal_val_images):
        img = img_to_array(load_img(str(img),
                                    target_size=(256, 256),
                                    color_mode='grayscale')) / 255
        val_data[i] = img
        val_labels[i] = 0
        
    # pneumonia cases
    for i, img in enumerate(pneumonia_val_images):
        img = img_to_array(load_img(str(img),
                                    target_size=(256, 256),
                                    color_mode='grayscale')) / 255
        val_data[i + normal_val_len] = img
        val_labels[i + normal_val_len] = 1

    return val_data, val_labels

val_data, val_labels = create_val_data(normal_val_len, pneumonia_val_len)

def create_test_data(normal_test_len, pneumonia_test_len):
    # get number of images
    test_len = normal_test_len + pneumonia_test_len

    # crete generators
    normal_test_images = glob.glob(normal_test_dir + '/*.jpeg')
    pneumonia_test_images = glob.glob(pneumonia_test_dir + '/*.jpeg')

    # create blank arrays
    test_data = np.zeros((test_len, 256, 256, 1), dtype=np.float32)
    test_labels = np.zeros((test_len, 1), dtype=np.int64)

    # normal cases
    for i, img in enumerate(normal_test_images):
        img = img_to_array(load_img(str(img),
                                    target_size=(256, 256),
                                    color_mode='grayscale')) / 255
        test_data[i] = img
        test_labels[i] = 0

    # pneumonia cases
    for i, img in enumerate(pneumonia_test_images):
        img = img_to_array(load_img(str(img),
                                    target_size=(256, 256),
                                    color_mode='grayscale')) / 255
        test_data[i + normal_test_len] = img
        test_labels[i + normal_test_len] = 1

    return test_data, test_labels

test_data, test_labels = create_test_data(normal_test_len, pneumonia_test_len)

#==================
# DATA AUGMENTATION
#==================
def random_rotate(img):
    # randomly selects rotation between -20% and 20%
    random_perc = random.uniform(-20, 20)
    return sk.transform.rotate(img, random_perc)

def horizontal_flip(img):
    # flip on y axis
    return img[:,::-1]

# dictionary of available transformation functions
available_transforms = {
    'rotate': random_rotate,
    'flip': horizontal_flip
}

random.seed(55)

train_data_aug = train_data.copy()
train_labels_aug = train_labels.copy()

# augment training data
for i, img in enumerate(train_data_aug):
    key = random.choice(list(available_transforms))
    trans_img = available_transforms[key](img)
    train_data_aug[i] = trans_img

# append original train_data to train_data_aug, then shuffle
train_data_aug = np.append(train_data_aug, train_data, axis=0)
train_labels_aug = np.append(train_labels_aug, train_labels, axis=0)
train_data_aug, train_labels_aug = shuffle(train_data_aug, 
                                           train_labels_aug,
                                           random_state=39)

#====================
# SAVE AUGMENTED DATA 
#====================
np.save('train_data_aug', train_data_aug)
np.save('train_labels_aug', train_labels_aug)