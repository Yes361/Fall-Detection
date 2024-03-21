import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
  min_detection_confidence = 0.85,
  min_tracking_confidence  = 0.85
)

specific_points = {
     'Head': mp_pose.PoseLandmark.NOSE, 
     'AnkleLeft': mp_pose.PoseLandmark.LEFT_ANKLE,
     'AnkleRight': mp_pose.PoseLandmark.RIGHT_ANKLE,
     'HipLeft': mp_pose.PoseLandmark.LEFT_HIP,
     'HipRight': mp_pose.PoseLandmark.RIGHT_HIP,
     'ShoulderLeft': mp_pose.PoseLandmark.LEFT_SHOULDER,
     'ShoulderRight': mp_pose.PoseLandmark.RIGHT_SHOULDER
  }