from mp_solutions import specific_points
import numpy as np
from kalman_filter import *
from screen_drawer import *

def get_bounding_box(frame,landmarks):
  xmax = 0
  ymax = 0
  xmin = frame.shape[1]
  ymin = frame.shape[0]
  for lm in landmarks.landmark:
     if lm.x < xmin: xmin = lm.x
     if lm.x > xmax: xmax = lm.x
     if lm.y < ymin: ymin = lm.y
     if lm.y > ymax: ymax = lm.y
  return (
     int(xmin),
     int(ymin),
     int(xmax),
     int(ymax)
  )
# positional threshold method
def z_coordinate_correction(frame, landmarks):
    top_y = frame.shape[0]
    bottom_y = 0
    for lm in landmarks:
      y = int(lm.y * frame.shape[0])
      if y < top_y:
          top_y = y
      if y > bottom_y:
          bottom_y = y
    
    return top_y,bottom_y, bottom_y - top_y

def points_beneath_thr(frame, landmarks, thr, dt, previous):
  ty = z_coordinate_correction(frame,landmarks)[0]
  return ty > frame.shape[0] - thr, [0]

# head-ankle method
def head_ankle_y_thr(frame, landmarks, thr, dt, previous):
  a_l_y = landmarks[specific_points['AnkleLeft']].y
  a_r_y = landmarks[specific_points['AnkleRight']].y
  h_y = landmarks[specific_points['Head']].y

  if (a_l_y == -1 and a_r_y == -1 or h_y == -1):
     return False, [-1]
  
  return (max(a_l_y, a_r_y) - h_y) * frame.shape[0] < thr, [(max(a_l_y, a_r_y) - h_y) * frame.shape[0]]

def velocity_thr(frame, landmarks, thr, dt, previous):
  nose_y = landmarks[specific_points['Head']].y
  l_ankle_y = landmarks[specific_points['AnkleLeft']].y
  r_ankle_y = landmarks[specific_points['AnkleRight']].y
  l_hip_y = landmarks[specific_points['HipLeft']].y
  r_hip_y = landmarks[specific_points['HipRight']].y
  vel = (nose_y - previous[4])/dt * frame.shape[0]
  return vel > thr, (l_ankle_y, r_ankle_y, l_hip_y, r_hip_y, nose_y, vel)

def acceleration_thr(frame, landmarks, thr, dt, previous):
  acceleration = get_acceleration(frame,landmarks)
  arrow(frame, landmarks[specific_points['Head']], acceleration['Head'])
  arrow(frame, landmarks[specific_points['AnkleLeft']], acceleration['AnkleLeft'])
  arrow(frame, landmarks[specific_points['AnkleRight']], acceleration['AnkleRight'])
  arrow(frame, landmarks[specific_points['HipLeft']], acceleration['HipLeft'])
  arrow(frame, landmarks[specific_points['HipRight']], acceleration['HipRight'])
  arrow(frame, landmarks[specific_points['ShoulderLeft']], acceleration['ShoulderLeft'])
  arrow(frame, landmarks[specific_points['ShoulderRight']], acceleration['ShoulderRight'])
  return acceleration['Head'][1] > thr, [acceleration['Head'][1]/10]

def acceleration_thr_v2(frame, landmarks, thr, dt, previous):
  acceleration = {}
  count = 0
  for name, KF in KF_points.items():
      state = KF.predict()
      acceleration[name] = (state[4,0],state[5,0])
      #print(state)
      KF.update(
        np.expand_dims(
        [
            landmarks[specific_points[name]].x * frame.shape[1], 
            landmarks[specific_points[name]].y * frame.shape[0]
        ], axis=-1))
  arrow(frame, landmarks[specific_points['Head']], acceleration['Head'])
  arrow(frame, landmarks[specific_points['ShoulderLeft']], acceleration['ShoulderLeft'])
  arrow(frame, landmarks[specific_points['HipLeft']], acceleration['HipLeft'])
  arrow(frame, landmarks[specific_points['HipRight']], acceleration['HipRight'])
  arrow(frame, landmarks[specific_points['ShoulderRight']], acceleration['ShoulderRight'])
  shoulder_center = (acceleration['ShoulderRight'][1] + acceleration['ShoulderLeft'][1])/2
  hip_center = (acceleration['HipLeft'][1] + acceleration['HipRight'][1])/2
  return acceleration['Head'][1] > thr, [acceleration['Head'][1]]

def acceleration_thr_v3(frame, landmarks, thr, dt, previous):
  acceleration = {}
  z = z_coordinate_correction(frame,landmarks)[2]
  for name, KF in KF_points.items():
      state = KF.predict()
      acceleration[name] = (state[4,0],state[5,0])
      #print(state)
      KF.update(
        np.expand_dims(
        [
            landmarks[specific_points[name]].x * frame.shape[1], 
            landmarks[specific_points[name]].y * frame.shape[0]
        ], axis=-1))
  arrow(frame, landmarks[specific_points['Head']], acceleration['Head'])
  return acceleration['Head'][1] > thr, z

def combo(frame, landmarks, thr, dt, previous):
   acceleration = previous[0] or acceleration_thr(frame, landmarks, thr[0], dt, previous)[0]
   points = previous[1] or points_beneath_thr(frame, landmarks, thr[1], dt, previous)[0]
   return acceleration and points, [acceleration, points]

fall_detection_methods = [points_beneath_thr, head_ankle_y_thr, velocity_thr, acceleration_thr, combo]