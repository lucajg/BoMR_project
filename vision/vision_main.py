import numpy as np
import cv2
import sys
from functions import *

# Define a class for circles
class Circle:
    def __init__(self, center, radius, color):
        self.center = center  # Center position as a tuple (x, y)
        self.radius = radius  # Radius as a float
        self.color = color    # Color as a tuple (B, G, R)

    def is_in_range(self, ref_color):
        ref_color = np.array(ref_color)
        dmax = 150 # max eculidian distance for the color to be considered red
        d = color_distance(self.color, ref_color)
        return (d <= dmax)

    def __repr__(self):
        return f"Circle(center={self.center}, radius={self.radius}, color={self.color})"

# Function to compute euclidian distance between two colors
def color_distance(color1, color2):
    color1 = np.array(color1)
    color2 = np.array(color2)
    return np.linalg.norm(color1 - color2)

# Function to compute the median point between two circles
def circles_med(circle1, circle2):
    x1, y1 = circle1.center
    x2, y2 = circle2.center

    median_x = (x1 + x2)/2
    median_y = (y1 + y2)/2

    return (median_x, median_y)

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
green_ref = (0,255,0)
red_ref = (0,0,255)


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
    
    frame = cv2.flip(frame, 1)

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
    if green_circle is not None:
        cv2.circle(frame, green_circle.center, green_circle.radius, green_ref, 3)
    if red_circle is not None:
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

print(traj)
source.release()
cv2.destroyWindow(win_name)