import numpy as np
import cv2

# Define a class for circles
class Circle:
    def __init__(self, center, radius, color):
        self.center = center  # Center position as a tuple (x, y)
        self.radius = radius  # Radius as a float
        self.color = color    # Color as a tuple (B, G, R)

    def isColorMatching(self, ref_color):
        # Color distance
        ref_color = np.array(ref_color)
        dmax = 80 # max eculidian distance for the color to be considered equivalent
        d = distance(self.color, ref_color)

        return d <= dmax

    def __repr__(self):
        return f"Circle(center={self.center}, radius={self.radius}, color={self.color})"
    

# Function to compute euclidian distance between two colors
def distance(vec1, vec2):
    vec1 = np.array(vec1, dtype=np.int32)  # Ensure values are floats
    vec2 = np.array(vec2, dtype=np.int32)
    return np.linalg.norm(vec1 - vec2)

# Function to compute the median point between two circles
def circles_med(circle1, circle2):
    x1, y1 = circle1.center
    x2, y2 = circle2.center

    median_x = (x1 + x2)/2
    median_y = (y1 + y2)/2

    return (median_x, median_y)

def robot_angle(green_center, red_center):
    red_center = np.array(red_center, dtype=np.int32)  # Ensure values are floats
    green_center = np.array(green_center, dtype=np.int32)
    x_r, y_r = red_center
    x_g, y_g = green_center
    
    # Calculate the direction vector
    dx = x_r - x_g
    dy = y_r - y_g
    
    # Compute the angle in radians
    angle_radians = np.arctan2(dy, dx)
    
    return angle_radians

def detect_squares_and_transform(frame):
    # Define the color range for pure RGB blue (BGR format in OpenCV)
    lower_blue = np.array([255, 0, 0])  # Lower bound for BGR
    upper_blue = np.array([255, 0, 0])  # Upper bound for BGR (pure blue)

    # Convert to HSV for better color segmentation
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue_hsv = np.array([110, 200, 200])  # Approx. HSV for bright blue
    upper_blue_hsv = np.array([130, 255, 255])

    # Threshold the image to isolate blue regions
    mask = cv2.inRange(hsv_frame, lower_blue_hsv, upper_blue_hsv)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours to detect squares
    squares = []
    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the contour is a square
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if 1000 < area < 10000:  # Filter by reasonable size
                squares.append(approx)

    # Ensure exactly 4 squares are detected
    if len(squares) != 4:
        print("Error: Expected 4 squares, but found", len(squares))
        return None

    # Sort squares based on their centroid to arrange in a specific order
    centroids = [np.mean(square.reshape(-1, 2), axis=0) for square in squares]
    sorted_indices = sorted(range(len(centroids)), key=lambda i: (centroids[i][1], centroids[i][0]))
    sorted_squares = [squares[i] for i in sorted_indices]

    # Extract coordinates for the 4 corners
    rectangle_corners = np.array([
        sorted_squares[0].reshape(-1, 2)[0],  # Top-left
        sorted_squares[1].reshape(-1, 2)[0],  # Top-right
        sorted_squares[2].reshape(-1, 2)[0],  # Bottom-right
        sorted_squares[3].reshape(-1, 2)[0]  # Bottom-left
    ], dtype=np.float32)

    # Define the destination rectangle
    rect_width = 400
    rect_height = 300
    destination_corners = np.array([
        [0, 0],  # Top-left
        [rect_width - 1, 0],  # Top-right
        [rect_width - 1, rect_height - 1],  # Bottom-right
        [0, rect_height - 1]  # Bottom-left
    ], dtype=np.float32)

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(rectangle_corners, destination_corners)

    # Perform the perspective transformation
    transformed_frame = cv2.warpPerspective(frame, matrix, (rect_width, rect_height))

    return transformed_frame