import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model('Braintumor10epochs.h5')


INPUT_SIZE = 64


st.title("Brain Tumor Classification")
st.write("Upload an image to predict whether it has a brain tumor or not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg"])

if uploaded_file is not None:
    
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=False)
    st.write("Classifying...")
    
    image = Image.open(uploaded_file)
    image = image.resize((INPUT_SIZE, INPUT_SIZE))  
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)  
    
    result = model.predict(input_img)
    
    if result[0][0] > 0.5:
        st.write("### Prediction: **Brain Tumor Detected**")
    else:
        st.write("### Prediction: **No Brain Tumor Detected**")
