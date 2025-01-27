import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse

def load_calib(calib_file):
    with open(calib_file,'r') as f:
        data = yaml.safe_load(f)

    camera_matrix = np.array(data['camera_matrix']['data']).reshape((3,3))
    dist_coeffs = np.array(data['distortion_coefficients']['data']).reshape((1,5))
    return camera_matrix, dist_coeffs

def parse_points(points_str):
    points_list = [tuple(map(float, point.strip().split(','))) for point in points_str.strip().split(';')]
    return np.array(points_list, dtype=np.float64)

def solve_pnp():

    parser = argparse.ArgumentParser(description='Solve PnP')
    parser.add_argument('--image', dest= 'image', type=str, required=True, help='path to the image containing aruco marker')
    parser.add_argument('--calib', dest='calibration_file', type=str, required=True, help='path to the calibration')
    parser.add_argument('--points2D', dest='points_2D', type=str, required=True, help='path to the u and v coordinate in numpy format')
    parser.add_argument('--point3d', dest='points_3D', type=str, required=True, help='path to the 3D world coordinate in numpy format')
    param = parser.parse_args()

    # Read Image MICROSCOPE
    im = cv2.imread(param.image)
    size = im.shape
    imageWidth = size[1]
    imageHeight = size[0]
    imageSize = [imageWidth, imageHeight]

    # Calibration parameters from the YAML file (MICROSCOPE)
    camera_matrix, dist_coeffs= load_calib(param.calibration_file)

    # Load points
    points_2D = parse_points(param.points_2D)
    points_3D = parse_points(param.points_3D)
    
    ######### SOLVE PNP ##########
    success, rvecs, tvecs = cv2.solvePnP(points_3D, points_2D, camera_matrix, dist_coeffs, flags = cv2.SOLVEPNP_ITERATIVE )
    print("rvec", rvecs)
    print("tvec", tvecs)

    #Test the solvePnP by projecting the 3D Points to camera
    projPoints = cv2.projectPoints(points_3D, rvecs, tvecs, camera_matrix, dist_coeffs)[0]

    for p in points_2D:
        cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1) 
    for p in projPoints:
        cv2.circle(im, (int(p[0][0]), int(p[0][1])), 3, (0,255,0), -1)

    cv2.imshow("image", im)
    cv2.waitKey(0)
    np_rodrigues = np.asarray(rvecs[:,:],np.float64)
    rmat = cv2.Rodrigues(np_rodrigues)[0]
    print("rotation", rmat)
    print("translation", tvecs)

if __name__ =='__main__':
    solve_pnp()

