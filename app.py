import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
import tempfile
from PIL import Image
import time

# Set Streamlit page config
st.set_page_config(
    page_title="Vehicle Detection in Adverse Weather",
    page_icon="🔍",
    layout="centered",
)

# --- App Header ---
st.markdown("<h1 style='text-align: center; color: #6C63FF;'>Vehicle Detection in Adverse Weather</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #555;'>Upload an Image or Provide a YouTube Link for Object Detection</h3>", unsafe_allow_html=True)

# Load YOLOv9t model
model = YOLO("tubes-viskom-yolov9t-best.pt")  # Use the correct YOLOv9t best model

# Function to perform object detection on images
def detect_image(image):
    results = model(image)
    return results[0].plot()  # Draw bounding boxes on the image

# Function to perform object detection on video and return frames
def detect_video(youtube_link):
    results = model(youtube_link, stream=True)  # Process video from YouTube link
    frames = [frame.plot() for frame in results]  # Collect the list of frames (images)
    return frames

# Function to convert frames to video
def frames_to_video(frames, output_path='output_video.mp4', fps=30):
    # Get the dimensions from the first frame
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # Codec for mp4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)  # Write each frame to the video

    out.release()  # Close the video file
    return output_path

# --- File Upload Section ---
st.sidebar.markdown("### Upload an Image or Provide YouTube Link")
uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
uploaded_video = st.sidebar.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
youtube_link = st.sidebar.text_input("Or Enter YouTube Video Link")

# --- Object Detection & Display Section ---
if uploaded_image:
    # Process Image
    st.markdown("### Detection from Image")
    image = Image.open(uploaded_image)
    st.image(image, caption="Original Image", use_column_width=True)

    # Object detection
    st.markdown("### Detected Objects in Image")
    with st.spinner("Processing image..."):
        detected_image = detect_image(np.array(image))
        st.image(detected_image, caption="Processed Image with Bounding Boxes", use_column_width=True)

elif uploaded_video:
    # Process Video from YouTube Link

    with open(f"./intermediate_video.mp4", "wb") as f:
        f.write(uploaded_video.getbuffer())

    st.markdown("### Detection from Video")
    with st.spinner("Processing video..."):
        annotated_frames = detect_video('./intermediate_video.mp4')
        
        # Convert frames to video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
            # video_path = temp_video_file.name
            video_path = 'output_video.mp4'
            frames_to_video(annotated_frames, output_path=video_path)

            # Display the video
            video_file = open(video_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)

elif youtube_link:
    # Process Video from YouTube Link
    st.markdown("### Detection from Youtube")
    st.text(f"Processing video from: {youtube_link}")
    with st.spinner("Processing video..."):
        annotated_frames = detect_video(youtube_link)
        
        # Convert frames to video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
            # video_path = temp_video_file.name
            video_path = 'output_video.mp4'
            frames_to_video(annotated_frames, output_path=video_path)

            # time.sleep(2)
            # Display the video
            video_file = open(video_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)

# --- Footer ---

# st.video('output_video.mp4')

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>"
    "Developed by Hawari and Dani"
    "</p>",
    unsafe_allow_html=True,
)
