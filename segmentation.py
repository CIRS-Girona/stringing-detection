import argparse
import cv2
import numpy as np

def empty(i):
    pass
# Callback function triggered when a trackbar value changes
def on_trackbar(val):

    # Get the current positions of all trackbars
    hue_min = cv2.getTrackbarPos("Hue Min", "TrackedBars")
    hue_max = cv2.getTrackbarPos("Hue Max", "TrackedBars")
    sat_min = cv2.getTrackbarPos("Sat Min", "TrackedBars")
    sat_max = cv2.getTrackbarPos("Sat Max", "TrackedBars")
    val_min = cv2.getTrackbarPos("Val Min", "TrackedBars")
    val_max = cv2.getTrackbarPos("Val Max", "TrackedBars")

    # Define the lower and upper bounds for HSV filtering
    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])

    # Create a mask by filtering the HSV image with the defined bounds
    imgMASK = cv2.inRange(imgHSV, lower, upper)

    cv2.imshow("Output1", img)
    cv2.imshow("Output2", imgHSV)
    cv2.imshow("Mask", imgMASK)

def segmentation_hsv():

    global img, imgHSV
    # Handle input parameters
    parser = argparse.ArgumentParser(description="Adjust HSV bounds to segment an image.")
    parser.add_argument('--image', dest='image_file', type=str, required=True, help='Path to the image file.')
    param = parser.parse_args()

    # Path to the image file
    path = param.image_file

    # Read the image from the specified path
    img = cv2.imread(path)
    if img is None:
        print(f"Error: Unable to load image from {path}.")
        return

    # Convert the image from BGR to HSV color space
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create a named window for the trackbars
    cv2.namedWindow("TrackedBars")
    cv2.resizeWindow("TrackedBars", 640, 240)

    # Create trackbars for adjusting HSV bounds
    cv2.createTrackbar("Hue Min", "TrackedBars", 0, 179, on_trackbar)
    cv2.createTrackbar("Hue Max", "TrackedBars", 179, 179, on_trackbar)
    cv2.createTrackbar("Sat Min", "TrackedBars", 0, 255, on_trackbar)
    cv2.createTrackbar("Sat Max", "TrackedBars", 255, 255, on_trackbar)
    cv2.createTrackbar("Val Min", "TrackedBars", 0, 255, on_trackbar)
    cv2.createTrackbar("Val Max", "TrackedBars", 255, 255, on_trackbar)

    # Initialize the display by calling the on_trackbar function once
    on_trackbar(0)

    # Wait until the user presses a key
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    segmentation_hsv()
