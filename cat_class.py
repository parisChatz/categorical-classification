import os
from random import shuffle

import cv2
import numpy as np
from matplotlib import pyplot
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# from keras.models import load_model
from tensorflow.keras.utils import plot_model
