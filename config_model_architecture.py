from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Activation
from config_hyperparameters import *

modelX = Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_size, img_size, 1),
           # kernel_initializer='normal',
           # kernel_regularizer=l2(l2_score),
           # bias_regularizer=l2(l2_score)
           ),
    # BatchNormalization(),
    # MaxPooling2D(),
    # Dropout(0.1),
    Conv2D(64, (3, 3), padding='same', activation='relu'
           # , kernel_initializer='normal',
           # kernel_regularizer=l2(l2_score),
           # bias_regularizer=l2(l2_score)
           ),
    # BatchNormalization(),
    MaxPooling2D(),
    # Dropout(0.1),
    Conv2D(128, (3, 3), padding='same', activation='relu'
           # , kernel_initializer='normal',
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


def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(img_size, img_size, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    return model
