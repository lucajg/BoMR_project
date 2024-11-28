import numpy as np
import cv2

# Open Camera
camera_index = 0
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

ref_color = (50, 60, 120)

is_running = True
while is_running:
    # Read frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    blur = cv2.bilateralFilter(frame,9,75,75)
    # Convert the frame to float for precise computation
    frame_float = blur.astype(np.float32)
    # Compute the Euclidean distance from the reference color
    ref_color = np.array(ref_color, dtype=np.float32)
    distance = np.sqrt(np.sum((frame_float - ref_color) ** 2, axis=-1))

    # Threshold the image to isolate blue regions
    dmax = 60
    mask = (distance <= dmax).astype(np.uint8) * 255

    cv2.imshow(win_name, mask)

    key = cv2.waitKey(1)
    if key in [ord('Q'), ord('q'), 27]:
        is_running = False

cap.release()
cv2.destroyWindow(win_name)