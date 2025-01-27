import serial
import cv2
import os
import time
import numpy as np
from pyspin import PySpin
import argparse



def collect_data():

    # Handle input parameters
    parser = argparse.ArgumentParser(description="collect data using microscope and FLIR cameras.")
    parser.add_argument('--serial_port', dest ='port',type=str, required=True, help='Communication interface between the computer and the printer machine (e.g., COM4)')
    parser.add_argument('--baud_rate', dest='baudrate', type=int, required=True, help='data transmission speed')
    parser.add_argument('--microscope_index', dest='micro_id', type=int, required=True, help='Index of the microscope connected to the computer')
    parser.add_argument('--cam_index', dest='cam_id',type=int, required=True, help='Index of the camera connected to the computer')
    parser.add_argument('--micro_dir', dest='out_micro_images_folder',type=str, required=True, help='Output folder containing the images taken by the microscope')
    parser.add_argument('--flir_dir', dest='out_camera_images_folder', type=str, required=True, help='Output folder containing the images taken by the camera')
    param = parser.parse_args()


    # Initialize serial connection
    ser = serial.Serial(port=param.port, baudrate=param.baudrate)

    # Initialize microscope camera
    microscope_camera = cv2.VideoCapture(param.micro_id)  

    # Initialize FLIR camera
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    flir_camera = cam_list.GetByIndex(param.cam_id)
    flir_camera.Init()

    # Directories to save images
    microscope_output_folder = param.out_micro_images_folder
    flir_output_folder = param.out_camera_images_folder
    os.makedirs(microscope_output_folder, exist_ok=True)
    os.makedirs(flir_output_folder, exist_ok=True)
    target_width, target_height = 1280, 720
    # Counter for photo filenames
    microscope_photo_counter = 1
    flir_photo_counter = 1

    # Whenever the data is available
    while True:
        
        # Read serial data
        data = ser.readline().decode().strip()
        # Check the received comment
        if data == "Pn1  Hola!":

            print("stop printer to capture foto")
            # Wait
            time.sleep(5)
            
            # Capture photo from microscope camera
            ret, frame = microscope_camera.read()
            if ret:
                microscope_photo_filename = os.path.join(microscope_output_folder, f"micro_image_{microscope_photo_counter}.jpg")
                cv2.imwrite(microscope_photo_filename, frame)
                print(f"Microscope photo captured and saved as {microscope_photo_filename}")
                microscope_photo_counter += 1
            else:
                print("Failed to capture photo from microscope camera.")
        
            # Capture photo from FLIR camera
            flir_camera.BeginAcquisition()
            image_result = flir_camera.GetNextImage()
            if image_result.IsIncomplete():
                print(f"FLIR image incomplete with image status {image_result.GetImageStatus()}")
            else:
                image_data = image_result.GetNDArray()
                undistorted_image = cv2.cvtColor(image_data, cv2.COLOR_BAYER_RG2RGB)
                height, width, _ = undistorted_image.shape
                start_x = (width - target_width) // 2
                #Crop the FLIR image to match the resolution of the microscope image
                cropped_image = undistorted_image[744:1936, start_x:start_x+target_width]
                flir_photo_filename = os.path.join(flir_output_folder, f"flir_image_{flir_photo_counter}.jpg")
                cv2.imwrite(flir_photo_filename, cropped_image)
                print(f"FLIR photo captured and saved as {flir_photo_filename}")
                flir_photo_counter += 1
                
                # Release image
                image_result.Release()
                flir_camera.EndAcquisition()
                
                # Resume print
                print("resume print")

if __name__ == '__main__':
    collect_data()
