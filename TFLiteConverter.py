"""
This script is used to convert the model to a tensorflow lite model.
"""

# Importing the libraries
import tensorflow as tf

# Loading the model
model_path = input("Enter the path to the model that you want to convert: ")
model = tf.keras.models.load_model(model_path)

print("Model loaded successfully!")

# Convert the model
print("Converting the model to a tensorflow lite model...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tf_lite_model = converter.convert()

# Save the model
with open("model.tflite", "wb") as f:
    f.write(tf_lite_model)
