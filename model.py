from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config as cfg


def get():
    model = Sequential([
        Conv2D(20, (4, 4), padding='same', activation='relu', use_bias=False, input_shape=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(40, (4, 4), padding='same', activation='relu', use_bias=False),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(60, (4, 4), padding='same', activation='relu', use_bias=False),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(3, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
