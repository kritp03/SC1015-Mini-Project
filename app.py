import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import plotly.express as px

json_path = "models/Model-3/model8166.json"
weights_path = "models/Model-3/model8166.h5"
with open(json_path, 'r') as json_file:
    model_json = json_file.read()

model = tf.keras.models.model_from_json(model_json)
model.load_weights(weights_path)

class_labels = ['neutral', 'sadness', 'happiness', 'surprise', 'anger', 'fear', 'contempt', 'disgust']
st.title("Facial Emotion Recognition Web App")
st.divider()

option = st.sidebar.selectbox("Select an option", ("Upload Image", "Capture from Camera"))

if option == "Upload Image":
    st.subheader("Upload an image for prediction")
    uploaded_image = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image = image.resize((48, 48)).convert('L')
        img_array = np.array(image)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]
        confidence = prediction[0][predicted_class_index]
        prediction_percentages = prediction[0] * 100

        st.write("##")
        st.subheader("Model Prediction")
        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}")

        st.write("##")
        st.subheader("Probability Distribution of Model Prediction")
        fig = px.bar(x=class_labels, y=prediction_percentages, labels={'x': 'Emotion Categories', 'y': 'Probability (%)'})
        fig.update_layout(xaxis_title="Emotions", yaxis_title="Probability (%)",
                    yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig, use_container_width=True)



elif option == "Capture from Camera":
    st.write("Camera Capture Mode")

    # cap = cv2.VideoCapture(0)  # 0 indicates the default camera

    # if not cap.isOpened():
    #     st.write("Error: Could not open camera.")
    #     st.stop()

    # stop_camera_button_key = "stop_camera_button"  # Unique key for the button

    # while True:
    #     ret, frame = cap.read()

    #     if not ret:
    #         st.write("Error: Could not read frame from camera.")
    #         break

    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     st.image(frame_rgb, channels="RGB", use_column_width=True, caption="Camera Capture")

    #     resized_frame = cv2.resize(frame_rgb, (224, 224)) / 255.0
    #     image = np.expand_dims(resized_frame, axis=0)

    #     prediction = model.predict(image)
    #     predicted_class_index = np.argmax(prediction)
    #     predicted_class = class_labels[predicted_class_index]
    #     confidence = prediction[0][predicted_class_index]

    #     st.write(f"Predicted Class: {predicted_class}")
    #     st.write(f"Confidence: {confidence:.2f}")  # Display confidence with 2 decimal places

    #     # Check for user input to stop camera capture
    #     if st.button("Stop Camera", key=stop_camera_button_key):
    #         break

    # cap.release()