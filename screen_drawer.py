import cv2
import numpy as np
from mp_solutions import *

drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
connection_spec = mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2)

def display_text(frame, word, coord):
  cv2.putText(
    frame,
    word,
    coord,
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
  )  

def draw_screen(frame,landmarks):
  connections = mp_pose.POSE_CONNECTIONS
  mp_drawing.draw_landmarks(
    frame, 
    landmarks, 
    connections, 
    connection_spec, 
    drawing_spec
    )
  
  for lm in landmarks.landmark:
    point(frame,(lm.x,lm.y))

def point(frame, coord):
  cv2.circle(
      frame, 
      (
        int(coord[0] * frame.shape[1]), 
        int(coord[1] * frame.shape[0])
      ), 
      drawing_spec.circle_radius, 
      drawing_spec.color,
      thickness = drawing_spec.thickness, 
      lineType = cv2.LINE_AA)

def mask(frame,thr):
  mask = np.zeros(frame.shape, dtype=np.uint8)
  mask[int(frame.shape[0] * thr):, :] = (0, 0, 255)
  opacity = 0.25
  cv2.addWeighted(mask, opacity, frame, 1-opacity, 0, frame)

def arrow(frame, coord, change):
  cv2.arrowedLine(frame,
                  (
                    int(coord.x * frame.shape[1]),
                    int(coord.y * frame.shape[0])
                  ),
                  (
                    int(coord.x * frame.shape[1] + change[0]), 
                    int(coord.y * frame.shape[0] + change[1])
                  ),
                  color=(0, 255, 0), thickness=2, tipLength=0.2)