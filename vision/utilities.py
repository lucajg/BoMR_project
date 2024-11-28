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

def detect_squares_and_transform(frame, ref_color):

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blur = cv2.bilateralFilter(frame_hsv,9,75,75)
    # Convert the frame to float for precise computation
    frame_float = frame_hsv.astype(np.float32)
    # Compute the Euclidean distance from the reference color
    ref_color = np.array(ref_color, dtype=np.float32)
    distance = np.sqrt(np.sum((frame_float - ref_color) ** 2, axis=-1))

    # Threshold the image to isolate blue regions
    dmax = 50
    mask = (distance <= dmax).astype(np.uint8) * 255

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Filter contours to detect squares
    squares = []
    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the contour is a square
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if 100 < area < 1000:  # Filter by reasonable size
                squares.append(approx)

    # Ensure exactly 4 squares are detected
    if len(squares) != 4:
        print("Error: Expected 4 squares, but found", len(squares))
    else:
        print("Succes: 4 squares found !")

    # Sort squares based on their centroid to arrange in a specific order
    centroids = []
    for square in squares:
        # Calculate centroid
        centroid = np.mean(square.reshape(-1, 2), axis=0)
        
        # Calculate edge length (assuming square shape and uniform edges)
        edge_length = np.linalg.norm(square[0] - square[1])  # Distance between the first two vertices
        
        # Append centroid and edge length
        centroids.append(np.append(centroid, edge_length))
    centroids = np.array(centroids)

    # Sort centroids

    y_sort = centroids[centroids[:, 1].argsort()]
    tops = y_sort[:2,:]
    bottoms = y_sort[2:,:]
    tops = tops[tops[:,0].argsort()]
    bottoms = bottoms[bottoms[:,0].argsort()]
    top_left = tops[0]
    top_left = top_left[:2] - top_left[2]/2
    top_right = tops[1]
    top_right = [top_right[0] + top_right[2]/2, top_right[1] - top_right[2]/2]
    bottom_left = bottoms[0]
    bottom_left = [bottom_left[0] - bottom_left[2]/2, bottom_left[1] + bottom_left[2]/2]
    bottoms_right = bottoms[1]
    bottoms_right = bottoms_right[:2] + bottoms_right[2]/2

    source_quad = np.array([top_left, top_right, bottom_left, bottoms_right], dtype=np.float32)

    # Define the destination rectangle
    rect_height, rect_width = frame.shape[:2]
    destination_rectangle = np.array([
        [0, 0],  # Top-left
        [rect_width, 0],  # Top-right
        [0, rect_height - 1],  # Bottom-left
        [rect_width - 1, rect_height - 1]  # Bottom-right
    ], dtype=np.float32)

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(source_quad, destination_rectangle)

    # Perform the perspective transformation
    transformed_frame = cv2.warpPerspective(frame, matrix, (rect_width, rect_height))

    return transformed_frame