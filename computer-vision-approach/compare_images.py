import cv2
import argparse

def compare_image():

    parser = argparse.ArgumentParser(description="Compare binary images.")
    parser.add_argument('--image1', dest='segmented_real_image', type=str, required=True, help='path to the binary images taken from real camera.')
    parser.add_argument('--image2', dest='segmented_blender_image',type=str, required=True, help='Path to the binary images taken from blender.')
    param = parser.parse_args()
 

    image1 = cv2.imread(param.segmented_real_image, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(param.segmented_blender_image, cv2.IMREAD_GRAYSCALE)

    # Check if images are loaded correctly
    if image1 is None:
        print(f"Error: Unable to load image from {param.segmented_real_image}.")
        return
    if image2 is None:
        print(f"Error: Unable to load image from {param.segmented_blender_image}.")
        return

    # Convert the images to BGR format
    image1_color = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    image2_color = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    image1_color[image1 > 128] = [0, 255, 0]
    image2_color[image2 > 128] = [0, 0, 255]

    # Overlay images
    both_images = cv2.addWeighted(image1_color, 0.5, image2_color, 0.5, 0)

    # Display the output
    cv2.imshow('image 1', image1_color)
    cv2.waitKey(0)
    cv2.imshow('image 2', image2_color)
    cv2.waitKey(0)
    cv2.imshow('overlaped images', both_images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    compare_image()


