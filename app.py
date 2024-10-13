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
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns  # Import seaborn for plotting

# =========================
# Global Variables
# =========================
FONTS = cv.FONT_HERSHEY_SIMPLEX
CENTER_THRESHOLD = 5
SIDE_THRESHOLD = 2
BLINK_THRESHOLD = 5
DISCOUNT_CENTER = 1
DISCOUNT_SIDE = 1
DISCOUNT_EYES = 20

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249,
          263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155,
           133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

mp_face_mesh = mp.solutions.face_mesh

# Initialize global variables for tracking
focus_score = 100
last_look_centered_time = None
not_looking_start_time = None
blink_start_time = None
total_blinks = 0
blink_detected = False
eyes_closed_start_time = None
# Add variables to track the last time we increased or decreased the focus score
last_focus_increase_time = None
last_focus_decrease_time = None

# =========================
# Helper Functions
# =========================

def euclidean_distance(point1, point2):
  return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

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
  iris_x, _ = iris_center

  if ratio > 5.5:
      return "Blink"
  elif iris_x < eye_left + hor_range * 0.3:
      return "Left"
  elif iris_x > eye_right - hor_range * 0.3:
      return "Right"
  else:
      return "Center"

def process_frame(frame, face_mesh, focus_score, last_look_centered_time, not_looking_start_time, blink_start_time, blink_detected, eyes_closed_start_time, last_focus_increase_time, last_focus_decrease_time):
  frame = cv.flip(frame, 1)
  rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
  results = face_mesh.process(rgb_frame)

  eye_direction_text = "Unknown"
  face_position = "Unknown"

  current_time = time.time()  # Moved here to ensure it's available for all branches

  if results.multi_face_landmarks:
      mesh_points = landmarks_detection(frame, results)
      
      # Face position monitoring
      face_3d = []
      face_2d = []
      for idx, lm in enumerate(results.multi_face_landmarks[0].landmark):
          if idx in [1, 33, 61, 199, 263, 291]:
              x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
              face_2d.append([x, y])
              face_3d.append([x, y, lm.z])
      
      face_2d = np.array(face_2d, dtype=np.float64)
      face_3d = np.array(face_3d, dtype=np.float64)

      focal_length = 1 * frame.shape[1]
      cam_matrix = np.array([[focal_length, 0, frame.shape[1] / 2],
                             [0, focal_length, frame.shape[0] / 2],
                             [0, 0, 1]])
      dist_matrix = np.zeros((4, 1), dtype=np.float64)
      success, rot_vec, trans_vec = cv.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
      rmat, jac = cv.Rodrigues(rot_vec)
      angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rmat)

      x = angles[0] * 360
      y = angles[1] * 360

      if y < -10:
          face_position = "Looking Left"
      elif y > 10:
          face_position = "Looking Right"
      elif x < -10:
          face_position = "Looking Down"
      elif x > 10:
          face_position = "Looking Up"
      else:
          face_position = "Forward"

      # Eye direction and blink detection
      ratio = blink_ratio(mesh_points, RIGHT_EYE, LEFT_EYE)
      left_iris_points = np.array([mesh_points[i] for i in LEFT_IRIS], dtype=np.int32)
      right_iris_points = np.array([mesh_points[i] for i in RIGHT_IRIS], dtype=np.int32)
      (l_cx, l_cy), l_radius = cv.minEnclosingCircle(left_iris_points)
      (r_cx, r_cy), r_radius = cv.minEnclosingCircle(right_iris_points)
      center_left = np.array([l_cx, l_cy], dtype=np.int32)
      center_right = np.array([r_cx, r_cy], dtype=np.int32)
      left_eye_direction = eye_direction(np.array([mesh_points[p] for p in LEFT_EYE]), center_left, ratio)
      right_eye_direction = eye_direction(np.array([mesh_points[p] for p in RIGHT_EYE]), center_right, ratio)

      if left_eye_direction == right_eye_direction:
          eye_direction_text = left_eye_direction
      else:
          eye_direction_text = left_eye_direction if left_eye_direction in ["Left", "Right"] else right_eye_direction

      # Focus scoring algorithm
      if face_position == "Forward" and eye_direction_text == "Center":
          if last_look_centered_time is None:
              last_look_centered_time = current_time
          not_looking_start_time = None
          if current_time - last_look_centered_time >= CENTER_THRESHOLD:
              # Increase focus score by 1% every 1 second when increasing
              if last_focus_increase_time is None or current_time - last_focus_increase_time >= 1:
                  focus_score = min(100, focus_score + 1)
                  last_focus_increase_time = current_time
      else:
          last_look_centered_time = None
          if not not_looking_start_time:
              not_looking_start_time = current_time
          elif current_time - not_looking_start_time >= SIDE_THRESHOLD:
              # Decrease focus score by 1% every 1 second when decreasing
              if last_focus_decrease_time is None or current_time - last_focus_decrease_time >= 1:
                  focus_score = max(0, focus_score - 1)
                  last_focus_decrease_time = current_time

      if ratio > 5.5:
          if not blink_detected:
              blink_start_time = current_time
              blink_detected = True
          elif current_time - blink_start_time >= BLINK_THRESHOLD:
              # Decrease focus score by 5% if eyes are closed for 5 seconds
              focus_score = max(0, focus_score - 5)
              blink_start_time = current_time
      else:
          if blink_detected:
              blink_detected = False

      # Display information on frame
      cv.putText(frame, f"Face: {face_position}", (50, 50), FONTS, 1, (255, 0, 0), 2, cv.LINE_AA)
      cv.putText(frame, f"Eyes: {eye_direction_text}", (50, 100), FONTS, 1, (0, 255, 0), 2, cv.LINE_AA)
      cv.putText(frame, f"Focus Score: {focus_score}%", (50, 150), FONTS, 1, (0, 0, 255), 2, cv.LINE_AA)

  else:
      # If no face is detected, decrease focus score by 1% every 1 second
      if last_focus_decrease_time is None or current_time - last_focus_decrease_time >= 1:
          focus_score = max(0, focus_score - 1)
          last_focus_decrease_time = current_time

  return (frame, focus_score, last_look_centered_time, not_looking_start_time, blink_start_time, blink_detected, eye_direction_text, face_position, last_focus_increase_time, last_focus_decrease_time)

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
  img = frame.to_ndarray(format="bgr24")
  global focus_score, last_look_centered_time, not_looking_start_time, blink_start_time, blink_detected, eyes_closed_start_time
  global last_focus_increase_time, last_focus_decrease_time  # Include new global variables
  global CENTER_THRESHOLD, SIDE_THRESHOLD, BLINK_THRESHOLD, DISCOUNT_SIDE, DISCOUNT_EYES

  with mp_face_mesh.FaceMesh(
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.7,
      min_tracking_confidence=0.7,
  ) as face_mesh:
      (img, focus_score, last_look_centered_time, not_looking_start_time, 
       blink_start_time, blink_detected, _, _, last_focus_increase_time, last_focus_decrease_time) = process_frame(
          img, face_mesh, focus_score, last_look_centered_time, not_looking_start_time, 
          blink_start_time, blink_detected, eyes_closed_start_time, last_focus_increase_time, last_focus_decrease_time
      )

  return av.VideoFrame.from_ndarray(img, format="bgr24")

def process_uploaded_video(video_file):
  tfile = tempfile.NamedTemporaryFile(delete=False) 
  tfile.write(video_file.read())
  
  cap = cv.VideoCapture(tfile.name)
  
  global focus_score, last_look_centered_time, not_looking_start_time, blink_start_time, blink_detected, eyes_closed_start_time
  # Initialize the new variables
  last_focus_increase_time = None
  last_focus_decrease_time = None

  focus_score = 100
  last_look_centered_time = None
  not_looking_start_time = None
  blink_start_time = None
  blink_detected = False
  eyes_closed_start_time = None
  
  data = []
  sleep_count = 0
  sleep_start = None
  total_blinks = 0
  start_time = None
  
  with mp_face_mesh.FaceMesh(
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.7,
      min_tracking_confidence=0.7,
  ) as face_mesh:
      while cap.isOpened():
          ret, frame = cap.read()
          if not ret:
              break
          
          (frame, focus_score, last_look_centered_time, not_looking_start_time, 
           blink_start_time, blink_detected, eye_direction, face_position, last_focus_increase_time, last_focus_decrease_time) = process_frame(
              frame, face_mesh, focus_score, last_look_centered_time, not_looking_start_time, 
              blink_start_time, blink_detected, eyes_closed_start_time, last_focus_increase_time, last_focus_decrease_time
          )
          
          timestamp = cap.get(cv.CAP_PROP_POS_MSEC) / 1000  # Convert to seconds
          
          if start_time is None:
              start_time = timestamp

          # Remove unknown data
          if eye_direction == "Unknown" or face_position == "Unknown":
              continue

          # Count total blinks
          if blink_detected:
              total_blinks += 1
          
          # Count continuous 10-second sleep intervals
          if eye_direction == "Blink" and face_position != "Forward":
              if sleep_start is None:
                  sleep_start = timestamp
              elif timestamp - sleep_start >= 10:
                  sleep_count += 1
                  sleep_start = None
          else:
              sleep_start = None
          
          data.append({
              'timestamp': timestamp,
              'focus_score': focus_score,
              'eye_direction': eye_direction,
              'face_position': face_position,
              'is_front_camera': face_position == "Forward" and eye_direction == "Center"
          })
  
  cap.release()
  df = pd.DataFrame(data)
  df['sleep_count'] = sleep_count
  df['total_blinks'] = total_blinks
  df['timestamp_min'] = (df['timestamp'] - start_time) / 60  # Convert to minutes

  # =========================
  # Updated Code Starts Here
  # =========================
  # Calculate Front Camera and Not Front Camera Time with the specified logic
  total_front_seconds = 0
  total_not_front_seconds = 0
  min_continuous_seconds = 60  # 1 minute

  current_state = None
  state_start_time = None

  for _, row in df.iterrows():
      state = 'front' if row['is_front_camera'] else 'not_front'
      if current_state is None:
          current_state = state
          state_start_time = row['timestamp']
      elif state != current_state:
          duration = row['timestamp'] - state_start_time
          if current_state == 'front':
              # Always add time for 'front' state
              total_front_seconds += duration
          else:
              # Only add time for 'not_front' state if duration exceeds 1 minute
              if duration >= min_continuous_seconds:
                  total_not_front_seconds += duration
          current_state = state
          state_start_time = row['timestamp']
      # If the state hasn't changed, we continue accumulating time in the current state

  # After iterating, handle the last accumulated state
  if current_state is not None:
      duration = df['timestamp'].iloc[-1] - state_start_time
      if current_state == 'front':
          total_front_seconds += duration
      else:
          if duration >= min_continuous_seconds:
              total_not_front_seconds += duration

  total_front_minutes = total_front_seconds / 60
  total_not_front_minutes = total_not_front_seconds / 60

  # Add these totals to the dataframe as metadata
  df.attrs['total_front_minutes'] = total_front_minutes
  df.attrs['total_not_front_minutes'] = total_not_front_minutes

  # =========================
  # Updated Code Ends Here
  # =========================

  return df

def create_dashboard(df):
  st.subheader("Individual Analytics")
  
  # Focus Score Trend
  fig_focus = go.Figure()
  fig_focus.add_trace(go.Scatter(x=df['timestamp_min'], y=df['focus_score'], mode='lines', name='Focus Score'))
  fig_focus.update_layout(title='Focus Score Over Time', xaxis_title='Time (minutes)', yaxis_title='Focus Score', yaxis_range=[0, 100])
  st.plotly_chart(fig_focus)
  
  # Front Camera Time vs Not Front Camera Time
  front_camera_time = df.attrs.get('total_front_minutes', 0)
  not_front_camera_time = df.attrs.get('total_not_front_minutes', 0)
  fig_camera = go.Figure(data=[go.Bar(x=['Front Camera', 'Not Front Camera'], y=[front_camera_time, not_front_camera_time], marker_color=['#636EFA', '#EF553B'])])
  fig_camera.update_layout(title='Front Camera Time vs Not Front Camera Time', xaxis_title='Camera Position', yaxis_title='Time (minutes)')
  st.plotly_chart(fig_camera)
  
  # Eye Direction Distribution
  eye_direction_counts = df['eye_direction'].value_counts()
  eye_direction_counts = eye_direction_counts[eye_direction_counts.index != 'Unknown']
  fig_eye = go.Figure(data=[go.Pie(labels=eye_direction_counts.index, values=eye_direction_counts.values)])
  fig_eye.update_layout(title='Eye Direction Distribution')
  st.plotly_chart(fig_eye)
  
  # Session Statistics
  st.subheader("Session Statistics")
  avg_focus_score = df['focus_score'].mean()
  st.write(f"Average Focus Score: {avg_focus_score:.2f}%")

def export_to_pdf(df):
  buffer = BytesIO()
  sns.set_style("whitegrid")  # Set seaborn style to include grids
  with PdfPages(buffer) as pdf:
      # Focus Score Over Time
      plt.figure(figsize=(10, 6))
      sns.lineplot(x='timestamp_min', y='focus_score', data=df)
      plt.title('Focus Score Over Time')
      plt.xlabel('Time (minutes)')
      plt.ylabel('Focus Score')
      plt.ylim(0, 100)
      pdf.savefig()
      plt.close()

      # Front Camera Time vs Not Front Camera Time
      front_camera_time = df.attrs.get('total_front_minutes', 0)
      not_front_camera_time = df.attrs.get('total_not_front_minutes', 0)
      plt.figure(figsize=(8, 6))
      sns.barplot(x=['Front Camera', 'Not Front Camera'], y=[front_camera_time, not_front_camera_time], palette=['#636EFA', '#EF553B'])
      plt.title('Front Camera Time vs Not Front Camera Time')
      plt.ylabel('Time (minutes)')
      pdf.savefig()
      plt.close()

      # Eye Direction Distribution
      eye_direction_counts = df['eye_direction'].value_counts()
      eye_direction_counts = eye_direction_counts[eye_direction_counts.index != 'Unknown']
      plt.figure(figsize=(8, 6))
      sns.barplot(x=eye_direction_counts.index, y=eye_direction_counts.values, palette='viridis')
      plt.title('Eye Direction Distribution')
      plt.ylabel('Count')
      plt.xlabel('Eye Direction')
      pdf.savefig()
      plt.close()

      # Blink Analysis (Added to PDF)
      plt.figure(figsize=(8, 6))
      total_blinks = df['total_blinks'].iloc[-1]
      total_duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
      total_minutes = total_duration / 60
      blinks_per_minute = total_blinks / total_minutes if total_minutes > 0 else 0
      sns.barplot(x=['Blinks per Minute'], y=[blinks_per_minute], palette='magma')
      plt.title('Blink Analysis')
      plt.ylabel('Blinks per Minute')
      pdf.savefig()
      plt.close()

      # Sleep Analysis (Added to PDF)
      plt.figure(figsize=(8, 6))
      sleep_count = df['sleep_count'].iloc[-1]
      if sleep_count > 0:
          plt.text(0.5, 0.5, f"Number of 10-second continuous sleep intervals: {sleep_count}", ha='center', va='center', fontsize=12)
      else:
          plt.text(0.5, 0.5, "Great job staying attentive throughout the session!", ha='center', va='center', fontsize=12)
      plt.title('Sleep Analysis')
      plt.axis('off')
      pdf.savefig()
      plt.close()

  buffer.seek(0)
  return buffer

def app():
  st.title("📊 Focus Detection with WebRTC and Video Analysis")

  # Sidebar for configuration
  st.sidebar.header("🔧 Configuration")
  global CENTER_THRESHOLD, SIDE_THRESHOLD, BLINK_THRESHOLD, DISCOUNT_SIDE, DISCOUNT_EYES

  # CENTER_THRESHOLD = st.sidebar.slider("Center Look Threshold (seconds)", 1, 10, 5, key="center_threshold")
  SIDE_THRESHOLD = st.sidebar.slider("Side Look Threshold (seconds)", 1, 10, 5, key="side_threshold")
  DISCOUNT_SIDE = st.sidebar.slider("Side Look Discount (%)", 1, 5, 1, key="discount_side")
  BLINK_THRESHOLD = st.sidebar.slider("Blink Threshold (seconds)", 1, 10, 5, key="blink_threshold")
  DISCOUNT_EYES = st.sidebar.slider("Closed Eyes Discount (%)", 5, 30, 5, key="discount_eyes")
  
  # Tabs for Live Video and Upload Video
  tab1, tab2 = st.tabs(["🎥 Live Video", "📤 Upload Video"])
  
  with tab1:
      st.header("🔴 Webcam Feed")
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
      st.header("📥 Upload Video for Analysis")
      uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
      
      if uploaded_file is not None:
          st.video(uploaded_file)
          
          if st.button("🔍 Analyze Video"):
              with st.spinner("Analyzing video..."):
                  results_df = process_uploaded_video(uploaded_file)
              
              st.success("✅ Analysis complete!")
              create_dashboard(results_df)
              
              # Export to PDF
              pdf_file = export_to_pdf(results_df)
              st.download_button(
                  label="💾 Download PDF Report",
                  data=pdf_file,
                  file_name="focus_analysis_report.pdf",
                  mime="application/pdf"
              )

if __name__ == "__main__":
  app()