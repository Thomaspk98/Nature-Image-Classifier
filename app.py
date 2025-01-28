# app.py
import streamlit as st
import tensorflow as tf
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing import image
import numpy as np
from PIL import Image

# Set the title of the app
st.title("Image Classification App")
st.write("Upload an image to classify it into one of the following categories: Mountain, Sea, Glacier, Street, Buildings, Forest.")

# Load the pre-trained model
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_my_model():
    return load_model('models/best_model.keras')  # Path to the trained model

model = load_my_model()

# Define the class labels
class_labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# Function to preprocess the image and make predictions
def predict_image(img):
    # Resize the image to match the model's input size
    img = img.resize((150, 150))
    # Convert the image to a numpy array and normalize it
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    # Make predictions
    predictions = model.predict(img_array)
    # Get the predicted class and confidence score
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption="Uploaded Image")

    # Make predictions
    predicted_class, confidence = predict_image(image_pil)
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")