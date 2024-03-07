import numpy as np
import pandas as pd
import cv2
import csv
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
from PIL import Image
import os

# Function to save predictions to a CSV file
def save_predictions(prediction, label, filename='emotions.csv'):
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['Prediction', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write headers if the file is empty
        if csvfile.tell() == 0:
            writer.writeheader()

        writer.writerow({'Prediction': prediction, 'Label': label})

# Load model and cascade classifiers
emotion_dict = {0:'angry', 1 :'happy', 2: 'neutral', 3:'sad', 4: 'surprise'}

json_file = open('emotion_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
classifier.load_weights("emotion_model1.h5")

try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

# Configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Class for face emotion detection
class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)

                label_position = (x, y)
                cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Save prediction to CSV file
                save_predictions(output, label='emotion_label')  # Replace 'emotion_label' with your label variable

        return img

# Function to display images in the sidebar
def display_images_in_sidebar():
    st.sidebar.title("Developer Team")
    folder_path = r"D:\new\ml_project\asset"  # Replace this with your folder path

    if not os.path.exists(folder_path):
        st.sidebar.error("Folder not found!")
        return

    image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    captions = [
        "Ahsan Ali",
        "Atif Malik",
              # Add more captions as needed for each image
    ]

    if not image_files:
        st.sidebar.warning("No images found in the folder.")
    else:
        for i in range(0, len(image_files), 2):
            col1, col2 = st.sidebar.columns(2)
            image1 = Image.open(image_files[i])
            caption1 = captions[i] if i < len(captions) else ""
            col1.image(image1, caption=caption1, use_column_width=True)

            if i + 1 < len(image_files):
                image2 = Image.open(image_files[i + 1])
                caption2 = captions[i + 1] if i + 1 < len(captions) else ""
                col2.image(image2, caption=caption2, use_column_width=True)

# Streamlit app
def main():
    st.title("Real Time Face Emotion Detection Application")
    activities = ["Home", "Webcam Face Detection", "Prediction View", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    
    st.sidebar.markdown(""" Developed by Ahsan Ali | Atif Malik  """)
    
    display_images_in_sidebar()  # Add the function call here

    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face Emotion detection application using OpenCV, Custom CNN model and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The key features of the application.

                 1. Real time face detection using web cam feed.

                 2. Real time face emotion recognization.

                 3. Saves the predictions of recognized emotions into a CSV file

                 """)
    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your face emotion")
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion)
    elif choice == "Prediction View":
        st.header("Saved Predictions")
        try:
            df = pd.read_csv('emotions.csv')
            st.write(df)
        except FileNotFoundError:
            st.write("No predictions found.")

    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Real time face emotion detection application using OpenCV, Custom Trained CNN model and Streamlit.</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                             		<div style="background-color:#98AFC7;padding:10px">
                             		<h4 style="color:white;text-align:center;">This Application is developed by Ahsan Ali & Atif Malik using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose. </h4>
                                    <h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass


if __name__ == "__main__":
    main()
