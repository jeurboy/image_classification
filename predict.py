#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import config as cfg
import model

import IPython.display as display
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)


checkpoint_dir = os.path.dirname(cfg.MODEL_PATH)
latest = tf.train.latest_checkpoint(checkpoint_dir)

model = model.get()
model.load_weights(latest)

model.summary()

validation_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our validation data

test_data_gen = validation_image_generator.flow_from_directory(
    directory=os.path.join(cfg.IMAGE_DIR, 'test'),
    batch_size=1,
    shuffle=True,
    target_size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
    class_mode=None,
    color_mode='rgb'
)
# augmented_images = [test_data_gen[0][0][0] for i in range(5)]
# plotImages(augmented_images)

# display([test_data_gen[0][0][0] for i in range(2)])

# predictions = model.predict(test_data_gen)
# print(predictions)

STEP_SIZE_TEST = test_data_gen.n//test_data_gen.batch_size
test_data_gen.reset()

pred = model.predict_generator(
    test_data_gen,
    steps=STEP_SIZE_TEST,
    verbose=1
)
predicted_class_indices = np.argmax(pred, axis=1)

print(pred)
print(predicted_class_indices)

filenames = test_data_gen.filenames
results = pd.DataFrame({
    "Filename": filenames,
    "Predictions": predicted_class_indices
})

print(results)
