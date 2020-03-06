import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# Arguments for data augmentation
from config_data_aug_params import data_gen_args


def plotImages(images_arr, img_size):
    for i in range(1, 7):
        plt.subplot(2, 3, i)
        plt.imshow(images_arr[i].reshape(img_size, img_size), cmap='gray')
        plt.tight_layout()
    plt.show()


def save_generated_images(train_X, train_Y):
    i = 0
    train_image_generator = ImageDataGenerator(**data_gen_args)  # Generator for our training data

    for batch in train_image_generator.flow(train_X, train_Y, batch_size=5, shuffle=True,
                                            save_to_dir="images/generated_images"):
        i += 1
        if i > 2:  # save 5 variations of 2 images
            break  # otherwise the generator would loop indefinitely


def plot_results(accuracy, val_accuracy, error, val_error, epoch, batch_size, learning_rate, opt, save_image=False):
    graph_name = 'images/documentation/{}/cat_vs_dog_metrics_plot_lr-{}_epochs-{}_batch-{}-{}.png'.format(opt,
                                                                                                          learning_rate,
                                                                                                          epoch,
                                                                                                          batch_size,
                                                                                                          '3conv-1base')
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
    plt.show(block=False)
    print('~~~~~~ Complete ~~~~~~')
    plt.pause(2)
    plt.close()
