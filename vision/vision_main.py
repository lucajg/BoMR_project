import numpy as np
import cv2
import sys
from functions import *


# Open Camera
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]


source = cv2.VideoCapture(s)

if not source.isOpened():
    print("Failed to open camera.")
else:
    print("Camera opened successfully.")

# Open window for visualization
win_name = "Camera"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

alive = True
while alive:
    has_frame, frame = source.read()
    if not has_frame:
        break
    
    result = cv2.Canny(frame, 40, 100)
    
    cv2.imshow(win_name, result)
    
    key = cv2.waitKey(1)
    if key == ord("Q") or key == ord("q") or key == 27:
        alive = False

source.release()
cv2.destroyWindow(win_name)