from tqdm import tqdm
import os
from cv2 import imread, resize, IMREAD_GRAYSCALE
from random import shuffle
import numpy as np


def label_image(img):
    # Input single image
    # Output returns label depending on name of image
    word_label = img.split('.')[-3]
    if word_label == "cat":
        return [1, 0]
    elif word_label == "dog":
        return [0, 1]


def process_data_set_grayscale(directory, dataset_type, img_size):
    # Input directory of dataset and state if it's "train_set" or "test" dataset
    # Output is an array of greyscaled images of the directory
    # It saves the array-ed dataset for future faster import of dataset

    data = []
    for img in tqdm(os.listdir(directory)):
        path = os.path.join(directory, img)
        label = label_image(img)
        img = resize(imread(path, IMREAD_GRAYSCALE), (img_size, img_size))
        data.append([np.array(img), np.array(label)])
    if dataset_type == "train":
        shuffle(data)
    np.save('{}_grayscale_data_{}x{}.npy'.format(dataset_type, img_size, img_size), data)
    return data


# new process data for rgb
def process_data_set_rgb(directory, dataset_type, img_size):
    data = []
    for img in tqdm(os.listdir(directory)):
        path = os.path.join(directory, img)
        label = label_image(img)
        img = resize(imread(path), (img_size, img_size))
        data.append([np.array(img), np.array(label)])
    if dataset_type == "train":
        shuffle(data)
    np.save('{}_rgb_data_{}x{}.npy'.format(dataset_type, img_size, img_size), data)
    return data


def analyse_images(train_dir, test_dir, val_dir):
    # Input directories
    # Output prints with number of data

    total_train_number = len(os.listdir(train_dir))
    total_test_number = len(os.listdir(test_dir))
    total_val_number = len(os.listdir(val_dir))

    print("--")
    print("Total training images:", total_train_number)
    print("Total validation images:", total_val_number)
    print("Total test images:", total_test_number)
    print("--")
    return total_train_number, total_val_number, total_test_number
