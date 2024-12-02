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
        dmax = 30 # max eculidian distance for the color to be considered equivalent
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

    # Extract map origin 
    # origin = bottom_left

    # Perform the perspective transformation
    transformed_frame = cv2.warpPerspective(frame, matrix, (rect_width, rect_height))

    return matrix

def frame2mapCoord(frame_coord):
    x, y = np.array(frame_coord)
    map_coord = np.array([-(y-427), x])
    return map_coord

def grid_centroids(grid_width, grid_height, frame_width, frame_height):
    # Compute centroids of the grid squares
    square_width = frame_width/grid_width
    square_height = frame_height/grid_height

    centroids = np.empty((0, 2), dtype=int)
    for row in range(grid_height):
        for col in range(grid_width):
            x = round((col + 0.5) * square_width)
            y = round((row + 0.5) * square_height)
            centroids = np.vstack((centroids, [x, y]))
    
    return centroids, square_width, square_height

def detect_obstacles(frame, centroids, square_width, square_height):
    # Frame preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Detect black squares
    obstacles = np.empty((0, 2), dtype=int)
    for centroid in centroids:
        x, y = centroid

        # Create a blank mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Define the square boundaries
        top_left = (int(x-square_width//2), int(y-square_height//2))
        bottom_right = (int(x+square_width//2), int(y+square_height//2))

        # Draw the region of interest on the mask
        cv2.rectangle(mask, top_left, bottom_right, 255, -1)

        # Compute mean value of the region of interest
        mean = cv2.mean(frame, mask=mask)[0]
        
        # Classify as obstacle or not
        if mean < 127:
            # Convert from pixel to grid position
            grid_x = round(x/square_width + 0.5)
            grid_y = round(y/square_height + 0.5)
            obstacles = np.vstack((obstacles, [grid_x, grid_y]))

        return obstacles
    
def detect_goal(frame, ref_color):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Convert the frame to float for precise computation
    frame_float = frame_hsv.astype(np.float32)
    # Compute the Euclidean distance from the reference color
    ref_color = np.array(ref_color, dtype=np.float32)
    distance = np.sqrt(np.sum((frame_float - ref_color) ** 2, axis=-1))

    # Threshold the image to isolate region of interest
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

    # Ensure exactly 1 squares are detected
    if len(squares) != 1:
        print("Error: Expected 1 squares, but found", len(squares))
        return False
    else:
        print("Succes: 1 squares found !")

    # Compute goal centroid
    square = squares[0]
    goal_centroid = np.mean(square.reshape(-1, 2), axis=0)
        

    return goal_centroid
    