import numpy as np
import cv2
import sys
from utilities import *
from time import sleep

PREVIEW  = 0  # Preview Mode
BLUR     = 1  # Blurring Filter
CANNY    = 2  # Canny Edge Detector

# Open Camera
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]


source = cv2.VideoCapture(s, cv2.CAP_DSHOW)

if not source.isOpened():
    print("Failed to open camera.")
else:
    print("Camera opened successfully.")

for i in range(100):
    source.read()

# Open window for visualization
win_name = "Camera"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

image_filter = PREVIEW

# Define global var
traj = np.empty((0, 2), dtype=int)
traj_angle = np.empty((0, 1), dtype=float)
green_ref = (40,60,30)
red_ref = (70,50,150)
circle_distance_max = 300


alive = True
while alive:
    # Initialize empty arrays
    circles = np.array([])
    green_circle_list = np.array([])
    red_circle_list = np.array([])

    # Reset detection flag
    detection = False
    tracking = False

    # Read frame from the camera
    ret, frame = source.read()
    if not ret:
        break
    
    #print('Resolution: ' + str(frame.shape[0]) + ' x ' + str(frame.shape[1]))

    # Preprocessing for Hough transform
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray,7)
    edge = cv2.Canny(blur, 75, 150)

    # Compute circle Hough transform
    circles_hough = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,1,param1=150,param2=20,minRadius=4,maxRadius=15)
    
    if circles_hough is not None:
        detection = True
        circles_hough = np.uint16(np.around(circles_hough))       
        print(len(circles_hough[0,:]),"detected")
        for i,circle in enumerate(circles_hough[0, :]):
            center = (circle[0], circle[1])
            radius = circle[2]
            # circle mean color
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            color = cv2.mean(frame, mask=mask)
            circles = np.append(circles, Circle(center, radius, color[:3]))
            print(circles[i], distance(circles[i].color, green_ref),distance(circles[i].color, red_ref))
            
    if detection and len(circles) > 1:
        # Keep track of green and red circles within radius range
        for circle in circles:
            # Check if the circle is green
            if circle.isColorMatching(green_ref):
                green_circle_list = np.append(green_circle_list, circle)
            # Check if the circle is red
            if circle.isColorMatching(red_ref):
                red_circle_list = np.append(red_circle_list, circle)

        # Extract circles closest to the ref colors
        green_circle = None
        red_circle = None
        min_distance = float('inf')  # Initialize with a large value
        # Green
        for circle in green_circle_list:
            color_distance = distance(circle.color, green_ref)
            if color_distance < min_distance:
                min_distance = color_distance
                green_circle = circle
        # Red
        min_distance = float('inf')  # Initialize with a large value
        for circle in red_circle_list:
            color_distance = distance(circle.color, red_ref)
            if color_distance < min_distance:
                min_distance = color_distance
                red_circle = circle
        #print(green_circle, red_circle)
        # Verify distance between circles
        if (green_circle is not None) and (red_circle is not None):
            #print("check 1",distance(green_circle.center, red_circle.center))
            if distance(green_circle.center, red_circle.center) < circle_distance_max:
                #print("check 2")
                tracking = True
                # Compute robot position and angle
                robot_pos = circles_med(green_circle, red_circle)
                heading_angle = robot_angle(green_circle.center, red_circle.center)
                # Convert to degrees
                heading_angle_degrees = np.degrees(heading_angle)
                
                # Normalize to [0, 360)
                heading_angle_degrees = (heading_angle_degrees + 360) % 360
                print(heading_angle_degrees)

                # Define text properties
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                text_color = (255, 255, 255)

                # Draw green and red circles
                cv2.circle(frame, green_circle.center, green_circle.radius, green_ref, 3)
                cv2.circle(frame, red_circle.center, red_circle.radius, red_ref, 3)

                # Print lables
                label = "GREEN"
                text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                text_x = green_circle.center[0] - text_size[0] // 2  # Center the text horizontally
                text_y = green_circle.center[1] - green_circle.radius - 10  # Place text above the circle
                cv2.putText(frame, label, (text_x, text_y), font, font_scale, text_color, font_thickness)
                label = "RED"
                text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                text_x = red_circle.center[0] - text_size[0] // 2  # Center the text horizontally
                text_y = red_circle.center[1] - red_circle.radius - 10  # Place text above the circle
                cv2.putText(frame, label, (text_x, text_y), font, font_scale, text_color, font_thickness)

                # Keep track of robot position
                robot_pos = (int(robot_pos[0]), int(robot_pos[1]))
                traj = np.vstack((traj, robot_pos))
                traj_angle = np.vstack((traj_angle, heading_angle))
                cv2.circle(frame, robot_pos, 1, (0,0,0),3)

    for i in range(1, len(traj)):
        cv2.line(frame, tuple(traj[i - 1]), tuple(traj[i]), (0, 255, 0), 2)

    if image_filter == PREVIEW:
        result = frame
    elif image_filter == CANNY:
        result = edge
    elif image_filter == BLUR:
        result = blur
    
    cv2.imshow(win_name, result)
    
    key = cv2.waitKey(1)
    if key == ord("A") or key == ord("a"):
        traj = np.empty((0, 2), dtype=int)
    if key == ord("Q") or key == ord("q") or key == 27:
        alive = False
    elif key == ord("C") or key == ord("c"):
        image_filter = CANNY
    elif key == ord("B") or key == ord("b"):
        image_filter = BLUR
    elif key == ord("P") or key == ord("p"):
        image_filter = PREVIEW
    sleep(0.1)

source.release()
cv2.destroyWindow(win_name)