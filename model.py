from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config as cfg


def get():
    model = Sequential([
        Conv2D(16, 5, padding='same', activation='relu', input_shape=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3)),
        MaxPooling2D(),
        Conv2D(32, 5, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 5, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(128, 5, padding='same', activation='relu'),
        MaxPooling2D(),
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
