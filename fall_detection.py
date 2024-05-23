import cv2
from screen_drawer import *
from fall_method import *
from mp_solutions import *
from kalman_filter import *
import time as t

import matplotlib.pyplot as plt

def main(option, method, thr, title):
  video = cv2.VideoCapture(option)

  # video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
  # video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

  initialize()
  begin_time = t.time()
  start = t.time()

  fall_time = t.time()
  previous = [0] * (len(specific_points)+1)
  fall = False
  fall_detected = False
  x = []
  y = []
  while video.isOpened():
    ret, frame = video.read()
    letter = cv2.waitKey(1) & 0xFF
    if letter == ord('w'):
      fall = False
    if (letter == ord('q')):
      break

    if t.time() - fall_time > 2000:
      fall = False
      fall_detected = False

    pose_points = pose.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).pose_landmarks
    dt = t.time() - begin_time
    if pose_points:
      joints = pose_points.landmark
      draw_screen(frame, pose_points)
      fall_detected, previous = method(frame, joints, thr, dt, previous)

      if t.time() - start < 1:
        continue
      #print(previous[1])

      x.append(t.time() - start)
      y.append(previous[0])

    if fall_detected or fall:
      display_text(frame,"Fall Detected!", (50, 50))
      
      if not fall:
        fall_time = t.time()
        
      fall = True
      #break
    begin_time = t.time()

    cv2.imshow(title, frame)
  #plt.plot(x,y)
  #plt.show()
  video.release()
  cv2.destroyAllWindows()
  return fall

if __name__ == "__main__":
  main(1, acceleration_thr, 100, "Fall Detection Algorithm")