import cv2
import numpy as np
import argparse


def morphological_operation():

    parser = argparse.ArgumentParser(description="compute stringing mask")
    parser.add_argument('--image1', dest='segmented_real_image', type=str, required=True, help='path to the binary images taken from real camera')
    parser.add_argument('--image2', dest='segmented_blender_image', type=str, required=True, help='path to the binary images taken from blender')
    parser.add_argument('--image3', dest='real_image', type=str, required=True, help='path to the original image taken from real camera')
    param = parser.parse_args()

    ###### FLIR #######
    real_segmented = cv2.imread(param.segmented_real_image, cv2.IMREAD_GRAYSCALE)
    blender_segmented = cv2.imread(param.segmented_blender_image, cv2.IMREAD_GRAYSCALE)
    real = cv2.imread(param.real_image)
    ## DEFINE THE KERNEl
    kernel = np.ones((11, 11), np.uint8)

    ## APPLY MORPHOLOGICAL OPERATION ON BLENDER IMAGE
    blender_dilated = cv2.dilate(blender_segmented, kernel)
    blender_erosion = cv2.erode(blender_segmented, kernel)
    boundry = cv2.bitwise_and(blender_dilated, cv2.bitwise_not(blender_erosion))

    ## APPLY MORPHOLOGICAL OPERATION ON REAL IMAGE
    closing_image = cv2.morphologyEx(real_segmented, cv2.MORPH_CLOSE, kernel,iterations =1)

    ## COMPUTE THE STRINGING MASK
    stringing_mask = cv2.bitwise_and(closing_image, cv2.bitwise_not(blender_dilated))

    ## CONVERT THE MASK TO RGB
    stringing_mask_color = cv2.cvtColor(stringing_mask, cv2.COLOR_GRAY2RGB)

    # Threshold the mask to get a binary mask
    _, binary_mask = cv2.threshold(stringing_mask_color, 127, 255, cv2.THRESH_BINARY)


    #Apply the stringing mask to the real image:
    real_stringing = real.copy()
    # Set the pixels inside the stringing mask to zero 
    real_stringing[binary_mask == 0] = 0


    # Show the result
    cv2.imshow('Original Image', real)
    cv2.waitKey(0)
    cv2.imshow('Original segmented', real_segmented)
    cv2.waitKey(0)
    cv2.imshow('blender dilated', blender_dilated)
    cv2.waitKey(0)
    cv2.imshow('Define edges', boundry)
    cv2.waitKey(0)
    cv2.imshow('closing image', closing_image)
    cv2.waitKey(0)
    cv2.imshow('stringing mask', stringing_mask)
    cv2.waitKey(0)
    cv2.imshow('stringing mask', real_stringing)
    cv2.waitKey(0)


    cv2.destroyAllWindows()


if __name__ == '__main__':
    morphological_operation()
