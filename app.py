import cv2 as cv
import mediapipe as mp
import numpy as np
import math
import time
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import av
import tempfile
import pandas as pd
import plotly.graph_objects as go

# Constants
FONTS = cv.FONT_HERSHEY_SIMPLEX
CENTER_THRESHOLD = 1     # Time threshold for looking at the center
SIDE_THRESHOLD   = 1     # Time threshold for looking to the side
BLINK_THRESHOLD  = 3     # Time threshold for closing eyes
DISCOUNT_CENTER  = 1     # Percentage discount for not looking at center
DISCOUNT_SIDE    = 1.5   # Percentage discount for looking to the side
DISCOUNT_EYES    = 20    # Percentage discount for closing eyes

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

mp_face_mesh = mp.solutions.face_mesh

def euclidean_distance(point, point1):
    return math.sqrt((point1[0] - point[0])**2 + (point1[1] - point[1])**2)**0.5

def blink_ratio(landmarks, right_indices, left_indices):
    rh_distance = euclidean_distance(landmarks[right_indices[0]], landmarks[right_indices[8]])
    rv_distance = euclidean_distance(landmarks[right_indices[12]], landmarks[right_indices[4]])
    lh_distance = euclidean_distance(landmarks[left_indices[0]], landmarks[left_indices[8]])
    lv_distance = euclidean_distance(landmarks[left_indices[12]], landmarks[left_indices[4]])

    if rv_distance == 0 or lv_distance == 0:
        return float('inf')

    re_ratio = rh_distance / rv_distance
    le_ratio = lh_distance / lv_distance
    return (re_ratio + le_ratio) / 2

def landmarks_detection(img, results):
    img_height, img_width = img.shape[:2]
    return [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]

def eye_direction(eye_points, iris_center, ratio):
    eye_left = np.min(eye_points[:, 0])
    eye_right = np.max(eye_points[:, 0])

    hor_range = eye_right - eye_left
    iris_x, iris_y = iris_center

    if ratio > 5.5:
        return "Blink"
    elif iris_x < eye_left + hor_range * 0.35:
        return "Left"
    elif iris_x > eye_right - hor_range * 0.35:
        return "Right"
    else:
        return "Center"

def process_frame(frame, face_mesh, focus_score, last_look_centered_time, not_looking_start_time, blink_start_time, total_blink_duration, blink_detected, eyes_closed_start_time):
    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    eye_direction_text = "Unknown"

    if results.multi_face_landmarks:
        mesh_points = landmarks_detection(frame, results)
        ratio = blink_ratio(mesh_points, RIGHT_EYE, LEFT_EYE)

        left_iris_points = np.array([mesh_points[i] for i in LEFT_IRIS], dtype=np.int32) #     may without numpy
        right_iris_points = np.array([mesh_points[i] for i in RIGHT_IRIS], dtype=np.int32)

        (l_cx, l_cy), l_radius = cv.minEnclosingCircle(left_iris_points)    #   l_radius   to   _
        (r_cx, r_cy), r_radius = cv.minEnclosingCircle(right_iris_points)
        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)

        left_eye_direction = eye_direction(np.array([mesh_points[p] for p in LEFT_EYE]), center_left, ratio)
        right_eye_direction = eye_direction(np.array([mesh_points[p] for p in RIGHT_EYE]), center_right, ratio)

        if left_eye_direction == right_eye_direction:   
            eye_direction_text = left_eye_direction

        else: # Mixed
            eye_direction_text = left_eye_direction if left_eye_direction in ["Left", "Right"] else right_eye_direction    



        if ratio > 5.5:
            if not blink_detected:
                blink_start_time = time.time()
                blink_detected = True
        else:
            if blink_detected:
                blink_duration = time.time() - blink_start_time
                total_blink_duration += blink_duration
                blink_detected = False

        current_time = time.time()
        if eye_direction_text == "Center":
            if last_look_centered_time is None:
                last_look_centered_time = current_time
            not_looking_start_time = None

            # check if he in center or not by check next frame - last in center
            # inside if still look to center
            if current_time - last_look_centered_time >= CENTER_THRESHOLD:
                focus_score = min(focus_score + DISCOUNT_CENTER, 100)
                total_blink_duration = 0
                
        else:
            last_look_centered_time = None
            if not not_looking_start_time:
                not_looking_start_time = current_time
            elif current_time - not_looking_start_time >= SIDE_THRESHOLD:
                focus_score -= DISCOUNT_SIDE
                not_looking_start_time = None

        if eyes_closed_start_time is not None and current_time - eyes_closed_start_time >= BLINK_THRESHOLD:
            focus_score -= DISCOUNT_EYES

        focus_score = max(0, min(focus_score, 100))

        cv.putText(frame, f"Eyes: {eye_direction_text}", (50, 100), FONTS, 1, (110, 240, 23), 2, cv.LINE_AA)
        cv.putText(frame, f"Focus Score: {focus_score}%", (50, 150), FONTS, 1, (110, 240, 23), 2, cv.LINE_AA)

    else:
        focus_score = 0

    return frame, focus_score, last_look_centered_time, not_looking_start_time, blink_start_time, total_blink_duration, blink_detected, eyes_closed_start_time, eye_direction_text

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    global focus_score, last_look_centered_time, not_looking_start_time, blink_start_time, total_blink_duration, blink_detected, eyes_closed_start_time

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as face_mesh:
        img, focus_score, last_look_centered_time, not_looking_start_time, blink_start_time, total_blink_duration, blink_detected, eyes_closed_start_time, _ = process_frame(
            img, face_mesh, focus_score, last_look_centered_time, not_looking_start_time, blink_start_time, total_blink_duration, blink_detected, eyes_closed_start_time
        )

    return av.VideoFrame.from_ndarray(img, format="bgr24")

def process_uploaded_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    
    cap = cv.VideoCapture(tfile.name)
    
    focus_score = 100
    last_look_centered_time = None
    not_looking_start_time = None
    blink_start_time = None
    total_blink_duration = 0
    blink_detected = False
    eyes_closed_start_time = None
    
    data = []
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame, focus_score, last_look_centered_time, not_looking_start_time, blink_start_time, total_blink_duration, blink_detected, eyes_closed_start_time, eye_direction = process_frame(
                frame, face_mesh, focus_score, last_look_centered_time, not_looking_start_time, blink_start_time, total_blink_duration, blink_detected, eyes_closed_start_time
            )
            
            data.append({
                'timestamp': cap.get(cv.CAP_PROP_POS_MSEC) / 1000,
                'focus_score': focus_score,
                'eye_direction': eye_direction
            })
    
    cap.release()
    return pd.DataFrame(data)

def create_dashboard(df):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['focus_score'], mode='lines', name='Focus Score'))
    
    fig.update_layout(
        title='Focus Score Over Time',
        xaxis_title='Time (seconds)',
        yaxis_title='Focus Score',
        yaxis_range=[0, 100]
    )
    
    st.plotly_chart(fig)
    
    eye_direction_counts = df['eye_direction'].value_counts()
    fig_pie = go.Figure(data=[go.Pie(labels=eye_direction_counts.index, values=eye_direction_counts.values)])
    fig_pie.update_layout(title='Eye Direction Distribution')
    
    st.plotly_chart(fig_pie)

def app():
    st.title("Focus Detection with WebRTC and Video Analysis")
    
    tab1, tab2 = st.tabs(["Live Video", "Upload Video"])
    
    with tab1:
        st.header("Webcam Feed")
        webrtc_streamer(
            key="camera",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={
                "video": True,
                "audio": False,
            },
            video_frame_callback=video_frame_callback,
        )
    
    with tab2:
        st.header("Upload Video for Analysis")
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            st.video(uploaded_file)
            
            if st.button("Analyze Video"):
                with st.spinner("Analyzing video..."):
                    results_df = process_uploaded_video(uploaded_file)
                
                st.success("Analysis complete!")
                create_dashboard(results_df)

if __name__ == "__main__":
    focus_score = 100
    last_look_centered_time = None
    not_looking_start_time = None
    blink_start_time = None
    total_blink_duration = 0
    blink_detected = False
    eyes_closed_start_time = None
    app()
