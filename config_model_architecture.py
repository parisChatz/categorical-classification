from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from config_hyperparameters import img_size
from tensorflow.keras.regularizers import l2


def define_model(l2_score):
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(img_size, img_size, 1),
               kernel_initializer='he_uniform',
               kernel_regularizer=l2(l2_score),
               bias_regularizer=l2(l2_score)
               ),
        # BatchNormalization(),
        # MaxPooling2D(),
        # Dropout(0.1),
        Conv2D(64, (3, 3), padding='same', activation='relu',
               kernel_initializer='he_uniform',
               kernel_regularizer=l2(l2_score),
               bias_regularizer=l2(l2_score)
               ),
        # BatchNormalization(),
        MaxPooling2D(),
        # Dropout(0.1),
        Conv2D(128, (3, 3), padding='same', activation='relu',
               kernel_initializer='he_uniform',
               kernel_regularizer=l2(l2_score),
               bias_regularizer=l2(l2_score)
               ),
        # BatchNormalization(),
        MaxPooling2D(),
        # Dropout(0.1),
        Flatten(),
        Dense(256, activation='relu',
              kernel_regularizer=l2(l2_score),
              bias_regularizer=l2(l2_score)
              ),
        # Dropout(0.3)
        Dense(2, activation='softmax')
    ])

    return model
