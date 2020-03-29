import os
import numpy as np
import pandas as pd

# Tensorflow libs
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, Nadam, RMSprop
# Load Model if Model saved
# from tensorflow.keras.models import load_model
# Save image of network
# from tensorflow.keras.utils import plot_model


# Custom libraries
import plotting
import data_preprocessing

# Custom config files
from config_data_aug_params import extreme_data_gen_args
from config_hyperparameters import *
from config_model_architecture import define_model

if __name__ == "__main__":

    # DEFINE IF IMAGES ARE GRAYSCALE color[0] ELSE color[1]
    im_color = color[0]

    best_test_acc = best_test_accuracy

    if os.path.exists('train_{}_data_{}x{}.npy'.format(im_color, img_size, img_size)):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Loading preprocessed dataset!")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        train = np.load('train_{}_data_{}x{}.npy'.format(im_color, img_size, img_size), allow_pickle=True)
        val = np.load('val_{}_data_{}x{}.npy'.format(im_color, img_size, img_size), allow_pickle=True)
        test = np.load('test_{}_data_{}x{}.npy'.format(im_color, img_size, img_size), allow_pickle=True)

        total_train, total_val, total_test = [len(train), len(val), len(test)]

    else:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("      Reading dataset!       ")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        if im_color is 'rgb':
            train = data_preprocessing.process_data_set_rgb(train_dir, "train", img_size)
            val = data_preprocessing.process_data_set_rgb(val_dir, "val", img_size)
            test = data_preprocessing.process_data_set_rgb(test_dir, "test", img_size)
        else:
            train = data_preprocessing.process_data_set_grayscale(train_dir, "train", img_size)
            val = data_preprocessing.process_data_set_grayscale(val_dir, "val", img_size)
            test = data_preprocessing.process_data_set_grayscale(test_dir, "test", img_size)

        # Show dataset shape*/w #
        total_train, total_val, total_test = data_preprocessing.analyse_images(train_dir, val_dir, test_dir)

    # # read whole Model
    # if os.path.exists('{}.meta'.format(model_name)):
    #     Model.load(model_name)
    #     print('Model loaded!')

    if im_color is 'grayscale':
        train_X = np.array([i[0] for i in train]).reshape(-1, img_size, img_size, 1)
        train_Y = [i[1] for i in train]
        val_X = np.array([i[0] for i in val]).reshape(-1, img_size, img_size, 1)
        val_Y = [i[1] for i in val]
        test_X = np.array([i[0] for i in test]).reshape(-1, img_size, img_size, 1)
        test_Y = [i[1] for i in test]
    else:
        # For RGB
        train_X = np.array([i[0] for i in train])
        train_Y = [i[1] for i in train]
        val_X = np.array([i[0] for i in val])
        val_Y = [i[1] for i in val]
        test_X = np.array([i[0] for i in test])
        test_Y = [i[1] for i in test]

    # Plot first 5 images from X array
    # plotting.plot_images(train_X[10:], img_size)

    # Initialise augmented image generators
    train_image_generator = ImageDataGenerator(**extreme_data_gen_args)  # Generator for our training data
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data
    test_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data

    # Generate images
    # Only needed if statistics are used(featurewise_center, featurewise_std_normalization, zca_whitening)
    if extreme_data_gen_args["featurewise_center"] or extreme_data_gen_args["featurewise_std_normalization"] is True:
        train_image_generator.fit(train_X)
        validation_image_generator.fit(test_X)

    train_data_gen = train_image_generator.flow(train_X, train_Y, batch_size=32, shuffle=True)
    val_data_gen = validation_image_generator.flow(val_X, val_Y, batch_size=32, shuffle=False)
    test_data_gen = test_image_generator.flow(test_X, test_Y, batch_size=32, shuffle=False)

    # Save generated images
    # plotting.save_generated_images(train_X, train_Y)

    # Network characteristics
    # Model.summary()

    metrics = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': [], 'test_acc': [], 'test_loss': [],
               'batch_size': [], 'optimizers': [], 'model': [], 'color': []}

    metrics = pd.DataFrame(data=metrics)

    batch = batch_size[0]
    epoch = epochs[0]

    for model in models:
        for optimizer in optimizers:

            model_name = '{}{}_batch-{}_opt-{}.h5'.format(model, color, batch_size, optimizers,
                                                          # learning_rate,
                                                          )

            modelX = define_model(model)  # ADD REG IF NEEDED

            opt1 = eval(optimizer)
            opt = opt1()  # Here put parameters for optimizer (learning_rate=learning_rate,etc)

            modelX.compile(optimizer=opt,
                           loss="categorical_crossentropy",
                           metrics=['categorical_accuracy'])

            history = modelX.fit(
                train_data_gen,
                steps_per_epoch=total_train // batch,
                validation_data=val_data_gen,
                validation_steps=total_val // batch,
                epochs=epoch)

            # # evaluate the model
            test_loss, test_accuracy = modelX.evaluate(test_data_gen, steps=len(test_data_gen), verbose=1)

            # Save image of network
            # plot_model(Model, to_file="images/documentation/" + model_name[:-3] + ".png",
            # show_shapes=True, expand_nested=True)

            # Load Model if Model saved
            # Model = tf.keras.models.load_model(model_name)
            # Model.summary()

            # Plot training & validation accuracy/loss values
            tr_acc = history.history['categorical_accuracy']
            val_acc = history.history['val_categorical_accuracy']

            tr_loss = history.history['loss']
            val_loss = history.history['val_loss']

            path = plotting.plot_results(tr_acc, val_acc, tr_loss, val_loss, epoch, batch,
                                         optimizer,
                                         model,
                                         im_color,
                                         # learning_rate,
                                         save_image=save_everything)

            # Save Model
            if test_accuracy >= best_test_acc and save_everything is True:
                modelX.save(path + 'best_model.h5')
            best_test_acc = test_accuracy

            temp_metrics = pd.Series(
                [tr_acc[-1], val_acc[-1], tr_loss[-1], val_loss[-1], test_accuracy, test_loss,
                 batch, optimizer, model, im_color],
                index=['train_acc', 'val_acc', 'train_loss', 'val_loss', 'test_acc', 'test_loss',
                       'batch_size', 'optimizers', 'model', 'color'])

            metrics = metrics.append(temp_metrics, ignore_index=True)

            if save_everything is True:
                metrics.to_csv(path + "metrics.csv", index=False)
