import cv2
import numpy as np


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