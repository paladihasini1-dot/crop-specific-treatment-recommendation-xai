import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model(
    r"C:\MAJOR PROJECT\rice_disease_model.keras"
)

# Disease classes
classes = [
    'bacterial_leaf_blight',
    'bacterial_leaf_streak',
    'bacterial_panicle_blight',
    'blast',
    'brown_spot',
    'dead_heart',
    'downy_mildew',
    'hispa',
    'normal',
    'tungro'
]

# Enter image path
img_path = input("Enter the path of rice leaf image: ")

# Load image
img = image.load_img(img_path, target_size=(224, 224))

# Convert image to array
img_array = image.img_to_array(img)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Prediction
predictions = model.predict(img_array)

# Get predicted class
predicted_index = np.argmax(predictions[0])
predicted_class = classes[predicted_index]

# Confidence
confidence = predictions[0][predicted_index] * 100

print("\n--------------------------------")
print("Rice Disease Prediction")
print("--------------------------------")
print("Disease:", predicted_class)
print("Confidence: {:.2f}%".format(confidence))
print("--------------------------------")