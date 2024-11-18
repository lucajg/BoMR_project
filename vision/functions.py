import cv2

def getCircles(frame):

    # Preprocess the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(frame, 5)

    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,
                              rows/8,param1=100,param2=30,minRadius=5,maxRadius=50)
    
    if circles is not None:
        return circles
    print("No circles found")
    return