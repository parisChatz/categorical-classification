from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from config_hyperparameters import img_size
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model


def define_model(name, l2_score, dropout=0):
    if name == "lenet5":
        model = Sequential([
            Conv2D(6, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(img_size, img_size, 1),
                   kernel_regularizer=l2(l2_score),
                   bias_regularizer=l2(l2_score)),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Dropout(dropout),
            Conv2D(16, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(l2_score),
                   bias_regularizer=l2(l2_score)),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Dropout(dropout),
            Flatten(),
            Dense(120, activation='relu', kernel_regularizer=l2(l2_score),
                  bias_regularizer=l2(l2_score)),
            Dropout(dropout),
            Dense(84, activation='relu', kernel_regularizer=l2(l2_score),
                  bias_regularizer=l2(l2_score)),
            Dropout(dropout),
            Dense(2, activation='softmax')
        ])
        return model
    elif name == "vgg1":
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                         input_shape=(img_size, img_size, 1), kernel_regularizer=l2(l2_score),
                         bias_regularizer=l2(l2_score)))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                         kernel_regularizer=l2(l2_score),
                         bias_regularizer=l2(l2_score)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(l2_score),
                        bias_regularizer=l2(l2_score)))
        model.add(Dropout(dropout))
        model.add(Dense(2, activation='softmax'))
        return model
    elif name == "vgg2":
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal', padding='same',
                         input_shape=(img_size, img_size, 1), kernel_regularizer=l2(l2_score),
                         bias_regularizer=l2(l2_score)))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='glorot_normal', padding='same',
                         kernel_regularizer=l2(l2_score),
                         bias_regularizer=l2(l2_score)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_normal', padding='same',
                         kernel_regularizer=l2(l2_score),
                         bias_regularizer=l2(l2_score)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='glorot_normal', padding='same',
                         kernel_regularizer=l2(l2_score),
                         bias_regularizer=l2(l2_score)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='glorot_normal', kernel_regularizer=l2(l2_score),
                        bias_regularizer=l2(l2_score)))
        model.add(Dropout(dropout))
        model.add(Dense(2, activation='softmax'))
        return model

    elif name == "vgg3":
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                         input_shape=(img_size, img_size, 1), kernel_regularizer=l2(l2_score),
                         bias_regularizer=l2(l2_score)))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                         kernel_regularizer=l2(l2_score),
                         bias_regularizer=l2(l2_score)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                         kernel_regularizer=l2(l2_score),
                         bias_regularizer=l2(l2_score)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                         kernel_regularizer=l2(l2_score),
                         bias_regularizer=l2(l2_score)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                         kernel_regularizer=l2(l2_score),
                         bias_regularizer=l2(l2_score)))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                         kernel_regularizer=l2(l2_score),
                         bias_regularizer=l2(l2_score)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(dropout))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(l2_score),
                        bias_regularizer=l2(l2_score)))
        model.add(Dropout(dropout))
        model.add(Dense(2, activation='softmax'))
        return model


def define_pretrained_model():
    # load model
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat_layer = Flatten()(model.layers[-1].output)
    dense_layer = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat_layer)
    output = Dense(2, activation='softmax')(dense_layer)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)

    return model
