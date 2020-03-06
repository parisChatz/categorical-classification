import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.regularizers import l2

# Custom libraries
import plotting
import data_preprocessing

# Custom config files
from config_data_aug_params import data_gen_args
from config_hyperparameters import *
from config_model_architecture import modelX as Model1
from config_model_architecture import define_model
from tensorflow.keras.optimizers import SGD
# Load Model if Model saved
# from keras.models import load_model

# Save image of network
from tensorflow.keras.utils import plot_model

# todo create other script with 4 models inside and import them one after the other

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

    # Plot first 5 images from X array
    # plotting.plotImages(train_X[10:], img_size)

    # todo develop a function that runs the whole trainning of a Model
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

    # todo for different batch sizes do epoch 200
    for lr in learning_rate:
        for batch in batch_size:
            for epoch in epochs:
                Model = define_model()
                # Compile

                # Adaptive Learning Rate SDG() (keep in mind adam works well with dropout)
                opt = SGD(learning_rate=lr)

                Model.compile(optimizer=opt,
                              loss="categorical_crossentropy",
                              metrics=["accuracy"])

                # Fit
                history = Model.fit(
                    train_data_gen,
                    steps_per_epoch=total_train // batch,
                    validation_data=test_data_gen,
                    validation_steps=total_test // batch,
                    epochs=epoch,
                    verbose=1)

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

                plotting.plot_results(acc, val_acc, loss, val_loss, epoch, batch, lr, "sgd", save_image=True)
