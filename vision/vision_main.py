import numpy as np
import cv2
import sys
from utilities import *
from time import sleep


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

# Define global var
traj = np.empty((0, 2), dtype=int)
green_ref = (50,120,50)
red_ref = (0,0,200)


alive = True
while alive:
    # Initialize empty arrays
    circles = np.array([])
    green_circles = np.array([])
    red_circles = np.array([])

    # Read frame from the camera
    has_frame, frame = source.read()
    if not has_frame:
        break
    
    frame = cv2.flip(frame, 1) # TO REMOVE WHEN USING EXT WEBCAM

    # Preprocessing for Hough transform
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # Compute circle Hough transform
    rows = gray.shape[0]
    circles_hough = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=5,maxRadius=50)
    
    if circles_hough is not None:
        circles_hough = np.uint16(np.around(circles_hough))       
        print(len(circles_hough[0,:]))
        for i,circle in enumerate(circles_hough[0, :]):
            center = (circle[0], circle[1])
            radius = circle[2]
            # circle mean color
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            color = cv2.mean(frame, mask=mask)
            circles = np.append(circles, Circle(center, radius, color[:3]))
            print(circles[i], color_distance(circles[i].color, green_ref),color_distance(circles[i].color, red_ref))
            
    
    # Keep track of green and red circles
    for circle in circles:
        # Check if the circle is green
        if circle.is_in_range(green_ref):
            green_circles = np.append(green_circles, circle)
        # Check if the circle is red
        if circle.is_in_range(red_ref):
            red_circles = np.append(red_circles, circle)

    # Extract circles closest to the ref colors
    green_circle = None
    red_circle = None
    min_distance = float('inf')  # Initialize with a large value
    # Green
    for circle in green_circles:
        distance = color_distance(circle.color, green_ref)
        
        if distance < min_distance:
            min_distance = distance
            green_circle = circle
    # Red
    min_distance = float('inf')  # Initialize with a large value
    for circle in red_circles:
        distance = color_distance(circle.color, red_ref)
        
        if distance < min_distance:
            min_distance = distance
            red_circle = circle

    # Compute position of the robot
    robot_pos = None
    if (green_circle is not None) and (red_circle is not None):
        robot_pos = circles_med(green_circle, red_circle)

    # Draw red and green circles
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    text_color = (255, 255, 255)
    if green_circle is not None:
        cv2.circle(frame, green_circle.center, green_circle.radius, green_ref, 3)
        label = "GREEN"
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        text_x = green_circle.center[0] - text_size[0] // 2  # Center the text horizontally
        text_y = green_circle.center[1] - green_circle.radius - 10  # Place text above the circle
        cv2.putText(frame, label, (text_x, text_y), font, font_scale, text_color, font_thickness)
    if red_circle is not None:
        label = "RED"
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        text_x = red_circle.center[0] - text_size[0] // 2  # Center the text horizontally
        text_y = red_circle.center[1] - green_circle.radius - 10  # Place text above the circle
        cv2.putText(frame, label, (text_x, text_y), font, font_scale, text_color, font_thickness)
        cv2.circle(frame, red_circle.center, red_circle.radius, red_ref, 3)
    if robot_pos is not None:
        robot_pos = (int(robot_pos[0]), int(robot_pos[1]))
        traj = np.vstack((traj, robot_pos))
        cv2.circle(frame, robot_pos, 1, (0,0,0),3)

    for i in range(1, len(traj)):
        cv2.line(frame, tuple(traj[i - 1]), tuple(traj[i]), (0, 255, 0), 2)

    cv2.imshow(win_name, frame)
    
    key = cv2.waitKey(1)
    if key == ord("Q") or key == ord("q") or key == 27:
        alive = False
    sleep(0.1)

source.release()
cv2.destroyWindow(win_name)