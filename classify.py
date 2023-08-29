import streamlit as st
import cv2
import numpy as np




def classify(image_array, model, class_names):
   
    # resize image to (150, 150)
    resized_image = cv2.resize(image_array, (150, 150))

    # convert image to numpy array
    image_array = np.asarray(resized_image)

    # normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # set model input
    data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
    data[0] = normalized_image_array


    prediction = model.predict(data)
    confidence_score = prediction[0][0]  
    class_index = 1 if confidence_score > 0.95 else 0
    class_name = class_names[class_index]

    return class_name, confidence_score
