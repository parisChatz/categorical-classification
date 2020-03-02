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

# Dataset variables
base_dir = "images"
train_dir = base_dir + '/train'
test_dir = base_dir + '/test'
img_size = 50  # 50x50 pixels
total_train = 0
total_test = 0

# Algorithm parameters
learning_rate = 1e-3
batch_size = 100
epochs = 150
l2_score = 1e-3
model_name = 'cat_vs_dog-{}--{}--{}.h5'.format(learning_rate, epochs, '3conv-1base')
graph_name = 'images/documentation/cat_vs_dog_metrics_plot_lr-{}_epochs-{}-{}.png'.format(learning_rate, epochs,
                                                                                          '3conv-1base')

model = Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_size, img_size, 1),
           # kernel_initializer='he_uniform',
           # kernel_regularizer=l2(l2_score),
           # bias_regularizer=l2(l2_score)
           ),
    # BatchNormalization(),
    # MaxPooling2D(),
    # Dropout(0.1),
    Conv2D(64, (3, 3), padding='same', activation='relu'
           # , kernel_initializer='he_uniform',
           # kernel_regularizer=l2(l2_score),
           # bias_regularizer=l2(l2_score)
           ),
    # BatchNormalization(),
    MaxPooling2D(),
    # Dropout(0.1),
    Conv2D(128, (3, 3), padding='same', activation='relu'
           # , kernel_initializer='he_uniform',
           # kernel_regularizer=l2(l2_score),
           # bias_regularizer=l2(l2_score)
           ),
    # BatchNormalization(),
    MaxPooling2D(),
    # Dropout(0.1),
    Flatten(),
    Dense(128, activation='relu',
          # kernel_regularizer=l2(l2_score),
          # bias_regularizer=l2(l2_score)
          ),
    # Dropout(0.2),
    Dense(2, activation='softmax')
])

# Arguments for data augmentation
data_gen_args = dict(rescale=1. / 255,
                     # featurewise_center=True,
                     # featurewise_std_normalization=True,
                     rotation_range=10,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     shear_range=0.2,
                     brightness_range=[0.8, 1.0]  # 0.5 unchanged, 0 all black 1, all white
                     )


def label_image(img):
    # Input single image
    # Output returns label depending on name of image
    word_label = img.split('.')[-3]
    if word_label == "cat":
        return [1, 0]
    elif word_label == "dog":
        return [0, 1]


def process_data_set(directory, dataset_type):
    # Input directory of dataset and state if it's "train" or "test" dataset
    # Output is an array of greyscaled images of the directory
    # It saves the array-ed dataset for future faster inport of dataset

    data = []
    for img in tqdm(os.listdir(directory)):
        label = label_image(img)
        path = os.path.join(directory, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (img_size, img_size))
        data.append([np.array(img), np.array(label)])
        # print(label,path)
    if dataset_type == "train":
        shuffle(data)
    np.save('{}_{}x{}_input_data.npy'.format(dataset_type, img_size, img_size), data)
    return data


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def analyse_images(traindir, testdir):
    # Input directories
    # Output prints with number of data

    total_train_number = len(os.listdir(traindir))
    total_test_number = len(os.listdir(testdir))

    print("--")
    print("Total training images:", total_train_number)
    print("Total validation images:", total_test_number)
    print("--")
    return total_train_number, total_test_number


def save_generated_images():
    i = 0
    for batch in train_image_generator.flow(train_X, train_Y, batch_size=25, shuffle=True,
                                            save_to_dir="images/generated_images"):
        i += 1
        if i > 5:  # save 25 variations of 5 images
            break  # otherwise the generator would loop indefinitely


def plot_results(accuracy, val_accuracy, error, val_error, epoch, save_image=False):
    epochs_range = range(epoch)
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accuracy, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, error, label='Training Loss')
    plt.plot(epochs_range, val_error, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    if save_image is True:
        plt.savefig(graph_name)
    plt.show()


# best pipeline? : best architecture -> best optimizer -> best LR -> best momentum
# Best optimizer
def find_optimizer(trainX, trainy, testX, testy):
    # create learning curves for different optimizers
    optimizer = ['sgd', 'rmsprop', 'adagrad', 'adam']
    for i in range(len(optimizer)):
        # determine the plot number
        plot_no = 220 + (i + 1)
        pyplot.subplot(plot_no)
        # fit model and plot learning curves for an optimizer
        # fit_model(trainX, trainy, testX, testy, optimizer[i])
    # show learning curves
    pyplot.show()


# Best learning rate
def find_learning_rate(trainX, trainy, testX, testy):
    # create learning curves for different learning rates
    learning_rates = [1E-0, 1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6, 1E-7]
    for i in range(len(learning_rates)):
        # determine the plot number
        plot_no = 420 + (i + 1)
        pyplot.subplot(plot_no)
        # fit model and plot learning curves for a learning rate
        # fit_model(trainX, trainy, testX, testy, learning_rates[i])
    # show learning curves
    pyplot.show()


# Best momentum
def find_momentum(trainX, trainy, testX, testy):
    # create learning curves for different momentums
    momentums = [0.0, 0.5, 0.9, 0.99]
    for i in range(len(momentums)):
        # determine the plot number
        plot_no = 220 + (i + 1)
        pyplot.subplot(plot_no)
        # fit model and plot learning curves for a momentum
        # fit_model(trainX, trainy, testX, testy, momentums[i])
    # show learning curves
    pyplot.show()


if __name__ == "__main__":
    if os.path.exists('train_data.npy'):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Loading preprocessed dataset!")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        train = np.load('train_data.npy', allow_pickle=True)
        test = np.load('test_data.npy', allow_pickle=True)
        total_train, total_test = [len(train), len(test)]

    else:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("      Reading dataset!       ")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        train = process_data_set(train_dir, "train")
        test = process_data_set(test_dir, "test")
        # Show dataset shape
        total_train, total_test = analyse_images(train_dir, test_dir)

    # # read whole model
    # if os.path.exists('{}.meta'.format(model_name)):
    #     model.load(model_name)
    #     print('model loaded!')

    train_X = np.array([i[0] for i in train]).reshape(-1, img_size, img_size, 1)
    train_Y = [i[1] for i in train]

    test_X = np.array([i[0] for i in test]).reshape(-1, img_size, img_size, 1)
    test_Y = [i[1] for i in test]

    # Plot first 5 images from X array
    # plotImages(train_X[:5])

    # Initialise augmented image generators
    train_image_generator = ImageDataGenerator(**data_gen_args)  # Generator for our training data
    validation_image_generator = ImageDataGenerator(**data_gen_args)  # Generator for our validation data

    # Generate images
    train_image_generator.fit(train_X)
    # validation_image_generator.fit(test_X)

    train_data_gen = train_image_generator.flow(train_X, train_Y, batch_size=25, shuffle=True)
    test_data_gen = validation_image_generator.flow(test_X, test_Y, batch_size=25, shuffle=False)

    # Save generated images
    # save_generated_images()

    # Network characteristics
    # model.summary()

    # Adaptive Learning Rate SDG() (keep in mind adam works well with dropout)
    opt = Adam()
    # opt = SGD(learning_rate=learning_rate)

    # Compile
    model.compile(optimizer=opt,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Fit
    history = model.fit(
        train_data_gen,
        steps_per_epoch=total_train // batch_size,
        validation_data=test_data_gen,
        validation_steps=total_test // batch_size,
        epochs=epochs)

    # Save image of network
    # plot_model(model, to_file="images/documentation/" + model_name[:-3] + ".png",
    # show_shapes=True, expand_nested=True)

    # Save model
    # model.save(model_name)
    # Load model if saved
    # model = tf.keras.models.load_model(model_name)
    # model.summary()

    # Plot training & validation accuracy/loss values
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # print('Train: %.3f, Test: %.3f' % (acc[-1], val_acc[-1]))
    plot_results(acc, val_acc, loss, val_loss, epochs, save_image=True)
