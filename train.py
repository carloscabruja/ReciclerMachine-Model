"""
This script is used to train the model.
"""

# Importing the libraries
import os
import shutil
import numpy as np
import tensorflow as tf


# DATA PREPROCESSING
print("DATA PREPROCESSING...")
print("---------------------")

accepted_images: list = os.listdir("data/accepted")
rejected_images: list = os.listdir("data/rejected")

print("Accepted images scanned: ", len(accepted_images))
print("Rejected images scanned: ", len(rejected_images))

# Downsampling the images
if len(accepted_images) > len(rejected_images):
    num_images_selected : int = len(rejected_images)
    accepted_images = np.random.choice(
        accepted_images, num_images_selected, replace=False
    )
else:
    num_images_selected : int = len(accepted_images)
    rejected_images = np.random.choice(
        rejected_images, num_images_selected, replace=False
    )

print(f"Num of images selected after downsampling: {num_images_selected}")
print("Creating dataset...")

# Ensuring that the dataset folders is empty
for filename in os.listdir("data/dataset/accepted"):
    os.remove(os.path.join("data/dataset/accepted", filename))

for filename in os.listdir("data/dataset/rejected"):
    os.remove(os.path.join("data/dataset/rejected", filename))

# Copying the images to the dataset folder
for i in range(num_images_selected):
    shutil.copy2(f"data/accepted/{accepted_images[i]}", "data/dataset/accepted")
    shutil.copy2(f"data/rejected/{rejected_images[i]}", "data/dataset/rejected")

print("Dataset created.")
print("Dataset accepted images: ", len(os.listdir("data/dataset/accepted")))
print("Dataset rejected images: ", len(os.listdir("data/dataset/rejected")))


# PREPARING THE MODEL
print("PREPARING THE MODEL...")
print("----------------------")
data_dir = "data/dataset"
batch_size = 64
img_height = 300
img_width = 300

# Creating the training and validation datasets
data_augmenter = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1.0 / 255,
    rotation_range=45,
    zoom_range=[0.5, 1.5],
    width_shift_range=0.25,
    height_shift_range=0.25,
    validation_split=0.2,
)

train_ds = data_augmenter.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True,
    subset="training",
)

val_ds = data_augmenter.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True,
    subset="validation",
)

InceptionV3_model = tf.keras.applications.InceptionV3(
    input_shape=(img_height, img_width, 3), include_top=False, weights="imagenet"
)
model = tf.keras.Sequential(
    [
        InceptionV3_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation="softmax"),
    ]
)

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()


# TRAINING THE MODEL
print("TRAINING THE MODEL...")
print("---------------------")

steps_per_epoch: int = train_ds.n // batch_size
validation_steps: int = val_ds.n // batch_size
epoch: int = 15

filepath = "models/model_{epoch:02d}_{val_accuracy:.2f}.h5"

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max"
)
call_backs = [checkpoint]


model.fit(
    train_ds,
    epochs=epoch,
    batch_size=batch_size,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    callbacks=call_backs,
)

print("Model trained.")
print("================")
print("END OF TRAINING.")
print("================")