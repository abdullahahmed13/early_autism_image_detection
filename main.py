import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
import cv2
from classify import classify

class_names = ["Not_Autistic", "Autistic"]


# set title
st.title('Early Stage Autism Spectrum Detection')

# set header
st.header('Please upload an Image For Your Child')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('model/autism_74.h5')


if file is not None:
    image = Image.open(file).convert('RGB')
    image_array = np.array(image)  # Convert PIL Image to NumPy array
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image_array, model, class_names)
    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
