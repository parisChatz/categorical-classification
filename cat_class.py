import os
import numpy as np
import pandas as pd

# Tensorflow libs
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
# Load Model if Model saved
# from tensorflow.keras.models import load_model
# Save image of network
from tensorflow.keras.utils import plot_model

# Custom libraries
import plotting
import data_preprocessing

# Custom config files
from config_data_aug_params import data_gen_args
from config_hyperparameters import *
from config_model_architecture import define_model

if __name__ == "__main__":
    if os.path.exists('train_data_{}x{}.npy'.format(img_size, img_size)):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Loading preprocessed dataset!")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        train = np.load('train_data_{}x{}.npy'.format(img_size, img_size), allow_pickle=True)
        test = np.load('test_data_{}x{}.npy'.format(img_size, img_size), allow_pickle=True)
        total_train, total_test = [len(train), len(test)]

    else:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("      Reading dataset!       ")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        train = data_preprocessing.process_data_set(train_dir, "train", img_size)
        test = data_preprocessing.process_data_set(test_dir, "test", img_size)
        # Show dataset shape
        total_train, total_test = data_preprocessing.analyse_images(train_dir, test_dir)

    # # read whole Model
    # if os.path.exists('{}.meta'.format(model_name)):
    #     Model.load(model_name)
    #     print('Model loaded!')

    train_X = np.array([i[0] for i in train]).reshape(-1, img_size, img_size, 1)
    train_Y = [i[1] for i in train]

    test_X = np.array([i[0] for i in test]).reshape(-1, img_size, img_size, 1)
    test_Y = [i[1] for i in test]

    # For RGBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
    # train_X = np.array([i[0] for i in train])
    # train_Y = [i[1] for i in train]
    # test_X = np.array([i[0] for i in test])
    # test_Y = [i[1] for i in test]

    # Plot first 5 images from X array
    # plotting.plot_images(train_X[10:], img_size)

    # Initialise augmented image generators
    train_image_generator = ImageDataGenerator(**data_gen_args)  # Generator for our training data
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data

    # Generate images
    # Only needed if statistics are used(featurewise_center, featurewise_std_normalization, zca_whitening)
    if data_gen_args["featurewise_center"] or data_gen_args["featurewise_std_normalization"] is True:
        train_image_generator.fit(train_X)
        validation_image_generator.fit(test_X)

    train_data_gen = train_image_generator.flow(train_X, train_Y, batch_size=10, shuffle=True)
    test_data_gen = validation_image_generator.flow(test_X, test_Y, batch_size=10, shuffle=False)

    # Save generated images
    # plotting.save_generated_images(train_X, train_Y)

    # Network characteristics
    # Model.summary()

    metrics = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': [], 'batch_size': [], 'optimizers': [],
               'regularizator': [], 'model': []}
    metrics = pd.DataFrame(data=metrics)

    for model in models:
        for optimizer in optimizers:
            # for learning_rate in learning_rates:
            for regularizator in l2_score:
                for batch in batch_size:
                    for epoch in epochs:
                        modelX = define_model(model, regularizator)

                        # opt1 = eval(optimizer)
                        # opt = opt1(learning_rate=learning_rate, momentum=momentum)

                        modelX.compile(optimizer=optimizer,
                                       loss="categorical_crossentropy",
                                       metrics=["accuracy"])

                        history = modelX.fit(
                            train_data_gen,
                            steps_per_epoch=total_train // batch,
                            validation_data=test_data_gen,
                            validation_steps=total_test // batch,
                            epochs=epoch)

                        # Save image of network
                        # plot_model(Model, to_file="images/documentation/" + model_name[:-3] + ".png",
                        # show_shapes=True, expand_nested=True)

                        # Save Model
                        # Model.save(model_name)

                        # Load Model if Model saved
                        # Model = tf.keras.models.load_model(model_name)
                        # Model.summary()

                        # Plot training & validation accuracy/loss values
                        acc = history.history['accuracy']
                        val_acc = history.history['val_accuracy']
                        loss = history.history['loss']
                        val_loss = history.history['val_loss']

                        # name = plotting.plot_results(acc, val_acc, loss, val_loss, epoch, batch, learning_rate,
                        #                              optimizers[0],
                        #                              regularizator,
                        #                              save_image=True)

                        name = plotting.plot_results_optimizers(acc, val_acc, loss, val_loss, epoch, batch,
                                                                optimizer, name,
                                                                save_image=True)

                        temp_metrics = pd.Series(
                            [acc[-1], val_acc[-1], loss[-1], val_loss[-1], batch, optimizer, regularizator, model],
                            index=['acc', 'val_acc', 'loss', 'val_loss', 'batch_size', 'optimizers',
                                   'regularizator', 'model'])

                        metrics = metrics.append(temp_metrics, ignore_index=True)
                        print(metrics.head())
                        metrics.to_csv("images/documentation/optimizer_metrics.csv", index=False)
