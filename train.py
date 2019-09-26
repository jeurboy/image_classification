from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# import pandas as pd
import config as cfg
import model
import tensorflow as tf
import os
import matplotlib.pyplot as plt

print(tf.__version__)


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column
batch_size = 64
epochs = cfg.EPOCHS
IMG_HEIGHT = cfg.IMAGE_SIZE
IMG_WIDTH = cfg.IMAGE_SIZE

train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    zoom_range=0.5
)  # Generator for our training data

validation_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our validation data

IMAGE_DIR = './images'
checkpoint_path = "model/1/cp_{epoch:04d}_cp.ckpt"

train_data_gen = train_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=os.path.join(IMAGE_DIR, 'train'),
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical',
    color_mode="rgb",
    seed=42
)

val_data_gen = validation_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=os.path.join(IMAGE_DIR, 'validation'),
    shuffle=False,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical',
    color_mode="rgb",
    # seed=42
)

checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1,
    save_freq=5  # Save every 5 epoch
)

model = model.get()

model.save_weights(checkpoint_path.format(epoch=0))

STEP_SIZE_TRAIN = train_data_gen.n//train_data_gen.batch_size
STEP_SIZE_VALID = val_data_gen.n//val_data_gen.batch_size

print("steps_per_epoch  : {:d} ".format(STEP_SIZE_TRAIN))
print("validation_steps : {:d} ".format(STEP_SIZE_VALID))

history = model.fit_generator(
    generator=train_data_gen,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=val_data_gen,
    validation_steps=STEP_SIZE_VALID,
    epochs=epochs
)

model.summary()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
