import numpy as np
import cv2
import sys
from utilities import *
from time import sleep

PREVIEW  = 0
BLUR     = 1
CANNY    = 2

# Open Camera
camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Failed to open camera.")
else:
    print("Camera opened successfully.")

for _ in range(100):
    cap.read()

# Open window for visualization
win_name = "Camera"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Initialize variables
image_filter = PREVIEW
positions = np.empty((0, 2), dtype=int)
angles = np.empty((0, 1), dtype=float)
GREEN_REF = (30,55,25)
MAG_REF_HSV = (172, 180, 120)
RED_REF = (40,35,120)
MAX_CIRCLE_DISTANCE = 300


cap_init = True
while cap_init:
    ret, frame = cap.read()
    if ret:
        cap_init = False

transformation_matrix = detect_squares_and_transform(frame, MAG_REF_HSV)

is_running = True
while is_running:
    # Reset per frames variables
    circles_detected = []
    green_circles = []
    red_circles = []
    robot_position = None
    robot_angle_deg = None
    detecting = False
    tracking = False

    # Read frame from the camera
    ret, frame = cap.read()
    if not ret:
        break
    height, width = frame.shape[:2]
    frame = cv2.warpPerspective(frame, transformation_matrix, (width, height))
    # Resizing with 3/2 aspect ratio
    frame = cv2.resize(frame, (640, 427))

    # Preprocessing for Hough transform
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray,7)
    edge = cv2.Canny(blur, 75, 150)

    # Compute circle Hough transform
    circles_hough = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1, minDist=1, 
        param1=150, param2=15, minRadius=2, maxRadius=15
    )
    
    if circles_hough is not None:
        circles_hough = np.uint16(np.around(circles_hough))       
        #print(len(circles_hough[0,:]),"detected")
        for i,circle in enumerate(circles_hough[0, :]):
            center = (circle[0], circle[1])
            radius = circle[2]
            # Mask for average color calculation
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            avg_color = cv2.mean(frame, mask=mask)[:3] #  alpha channel not useful
            circles_detected.append(Circle(center, radius, avg_color))
            #print(circles_detected[i], distance(circles_detected[i].color, GREEN_REF),distance(circles_detected[i].color, RED_REF))
            
        if len(circles_detected) > 1:
            # Keep track of green and red
            for circle in circles_detected:
                # Check if the circle is green
                if circle.isColorMatching(GREEN_REF):
                    green_circles.append(circle)
                # Check if the circle is red
                elif circle.isColorMatching(RED_REF):
                    red_circles.append(circle)

            # Find the closest green and red circles to their references
            green_circle = min(green_circles, key=lambda c: distance(c.color, GREEN_REF), default=None)
            red_circle = min(red_circles, key=lambda c: distance(c.color, RED_REF), default=None)
            #print(green_circle, red_circle)

            # Validate and calculate robot position and orientation
            if green_circle and red_circle:
                dist_between_circles = distance(green_circle.center, red_circle.center)
                if dist_between_circles < MAX_CIRCLE_DISTANCE:
                    tracking = True

                    # Compute robot position and angle
                    robot_position = circles_med(green_circle, red_circle)
                    robot_angle_rad = robot_angle(green_circle.center, red_circle.center)

                    # Convert to degrees and normalize to [0, 360)
                    heading_angle_degrees = (np.degrees(robot_angle_rad) + 360) % 360

                    # Draw green and red circles
                    cv2.circle(frame, green_circle.center, green_circle.radius, GREEN_REF, 3)
                    cv2.circle(frame, red_circle.center, red_circle.radius, RED_REF, 3)

                    # Define text properties
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    font_thickness = 2
                    text_color = (255, 255, 255)

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

                    # Update trajectory
                    robot_position = tuple(map(int, robot_position))
                    robot_position_map_coord = frame2mapCoord(robot_position)
                    positions = np.vstack((positions, robot_position))
                    angles  = np.vstack((angles, robot_angle_deg))
                    cv2.circle(frame, robot_position, 3, (0,0,0),-1)

    for i in range(1, len(positions)):
        cv2.line(frame, tuple(positions[i - 1]), tuple(positions[i]), (0, 255, 0), 2)

    if image_filter == PREVIEW:
        result = frame
    elif image_filter == CANNY:
        result = edge
    elif image_filter == BLUR:
        result = blur
    
    cv2.imshow(win_name, result)
    
    key = cv2.waitKey(1)
    if key in [ord("A"), ord("a")]:
        positions = np.empty((0, 2), dtype=int)
    if key in [ord('Q'), ord('q'), 27]:
        is_running = False
    elif key in [ord("C"), ord("c")]:
        image_filter = CANNY
    elif key in [ord("B"), ord("b")]:
        image_filter = BLUR
    elif key in [ord("P"), ord("p")]:
        image_filter = PREVIEW
    
    sleep(0.1)

cap.release()
cv2.destroyWindow(win_name)