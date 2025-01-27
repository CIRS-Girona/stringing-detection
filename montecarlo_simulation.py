import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
import yaml


########## get Random Gaussian noise ##############
def add_noise(mean, std_dev):
    noise_u = np.random.normal(mean, std_dev)
    noise_v = np.random.normal(mean, std_dev)
    return noise_u, noise_v


########## Define ellipsoid error ##################
def get_cov_ellipsoid(cov, mu=np.zeros((3))):
    """
    Return the 3D points representing the covariance matrix
    cov centred at mu and scaled by the factor nstd.
    """
    assert cov.shape==(3,3)

    # Find and sort eigenvalues to correspond to the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.sum(cov,axis=0).argsort()
    eigvals_temp = eigvals[idx]
    idx = eigvals_temp.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]

    # Set of all spherical angles to draw  ellipsoid
    n_points = 100
    theta = np.linspace(0, 2*np.pi, n_points)
    phi = np.linspace(0, np.pi, n_points)

    # Width, height and depth of ellipsoid
    rx, ry, rz = np.sqrt(7.815) * np.sqrt(eigvals)

    # Get the xyz points for plotting
    # Cartesian coordinates that correspond to the spherical angles:
    X = rx * np.outer(np.cos(theta), np.sin(phi))
    Y = ry * np.outer(np.sin(theta), np.sin(phi))
    Z = rz * np.outer(np.ones_like(theta), np.cos(phi))

    # Rotate ellipsoid for off-axis alignment
    old_shape = X.shape

    # Flatten to vectorize rotation
    X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
    X,Y,Z = np.matmul(eigvecs, np.array([X,Y,Z]))
    X,Y,Z = X.reshape(old_shape), Y.reshape(old_shape), Z.reshape(old_shape)

    # Add in offsets for the mean
    X = X + mu[0]
    Y = Y + mu[1]
    Z = Z + mu[2]
    
    return X,Y,Z


def parse_points(points_str):
    points_list = [tuple(map(float, point.strip().split(','))) for point in points_str.strip().split(';')]
    return np.array(points_list, dtype=np.float64)

def load_calibration(calibration_file):
    with open(calibration_file, "r") as f:
        data = yaml.safe_load(f)

    camera_matrix = np.array(data['camera_matrix']['data']).reshape((3,3))
    dist_coeffs = np.array(data['distortion_coefficients']['data']).reshape((1,5))
    return camera_matrix, dist_coeffs

def montecarlo_simulation():

    # Handle input parameters
    parser = argparse.ArgumentParser(description="Montecarlo simulation")
    parser.add_argument('--image', dest= 'image', type=str, required=True, help='path to the image containing aruco marker')
    parser.add_argument('--calib', dest='calibration_file', type=str, required=True, help='path to the calibration')
    parser.add_argument('--points2D', dest='points_2D', type=str, required=True, help='path to the u and v coordinate in numpy format')
    parser.add_argument('--point3d', dest='points_3D', type=str, required=True, help='path to the 3D world coordinate in numpy format')
    param = parser.parse_args()

    
    # Load calibration parameters from the YAML file
    camera_matrix, dist_coeffs = load_calibration(param.calibration_file)

    # Read Image
    im = cv2.imread(param.image)
    size = im.shape
    imageWidth = size[1]
    imageHeight = size[0]
    imageSize = [imageWidth, imageHeight]

    # Load points
    points_2D = parse_points(param.points_2D)
    points_3D = parse_points(param.points_3D)

    '''
    ###### u & v coordinate ##############
    points_2D = np.array([
        (763, 311),(367, 367),(503, 307),
        (895, 371),(761, 564),(349, 492),
        (902, 497),(481, 561)
        ], dtype=np.float64)

    ######## 3D world coordinae ##########
    points_3D = np.array([
        (0.1249, 0.135,  0),(0.1004,  0.125,  0),(0.1085,  0.1346,  0),
        (0.1328, 0.1274, 0),(0.124, 0.102, 0),(0.10,  0.109,  0),
        (0.1325, 0.111, 0),(0.108,  0.1013,  0)
        ],dtype=np.float32)
    '''

    # Parameters for noise
    noiseMean = 0
    noiseStd = 1

    camera_locations = []
    camera_locations_1 = []

    # Main loop
    for numberOfMarkers in [3,6,7,8]:  # numberOfMarkers from 3 to 8
        for iteration in range(1000):  # Monte-Carlo iterations
            noisy_points_2D = points_2D[:numberOfMarkers].copy()
            for markerID in range(numberOfMarkers):
                noise_u, noise_v = add_noise(noiseMean, noiseStd)
                noisy_points_2D[markerID] += [noise_u, noise_v]
        
            # Solve PnP
            if numberOfMarkers == 3:
                success, rvec, tvec = cv2.solveP3P(points_3D[:3], noisy_points_2D[:3], camera_matrix, dist_coeffs, flags = cv2.SOLVEPNP_P3P)
                success1, rvec1, tvec1 = cv2.solveP3P(points_3D[:3], points_2D[:3], camera_matrix, dist_coeffs,flags=cv2.SOLVEPNP_P3P)
                if success:
                    # Convert rotation vector to matrix
                    rvec_data, _ = rvec 
                    rmat, _ = cv2.Rodrigues(rvec_data)
                    tvec,_= tvec 
            
                    # Create a homogeneous transformation matrix
                    tmat = np.hstack((rmat, tvec))
                    tmat = np.vstack((tmat, [0, 0, 0, 1]))

                    # Extract camera location (inverse of the transformation matrix)
                    camera_location = np.linalg.inv(tmat)[:3, 3]
                    camera_locations.append(camera_location)

                if success1:

                    # Convert rotation vector to matrix
                    rvec_data_1, _ = rvec1
                    rmat1, _ = cv2.Rodrigues(rvec_data_1)

                    # Create a homogeneous transformation matrix
                    tvec1,_= tvec1
                    tmat1 = np.hstack((rmat1, tvec1))
                    tmat1 = np.vstack((tmat1, [0, 0, 0, 1]))

                    # Extract camera location (inverse of the transformation matrix)
                    camera_location_1 = np.linalg.inv(tmat1)[:3, 3]
                    camera_locations_1.append(camera_location_1)
            else:
                success, rvec, tvec = cv2.solvePnP(points_3D[:numberOfMarkers], noisy_points_2D[:numberOfMarkers], camera_matrix, dist_coeffs,flags=cv2.SOLVEPNP_ITERATIVE)
                success1, rvec1, tvec1 = cv2.solvePnP(points_3D[:numberOfMarkers], points_2D[:numberOfMarkers], camera_matrix, dist_coeffs,flags=cv2.SOLVEPNP_ITERATIVE)

                if success:

                    # Convert rotation vector to matrix
                    rmat,_ = cv2.Rodrigues(rvec)

                    # Create a homogeneous transformation matrix
                    tmat = np.hstack((rmat, tvec))
                    tmat = np.vstack((tmat, [0, 0, 0, 1]))

                    # Extract camera location (inverse of the transformation matrix)
                    camera_location = np.linalg.inv(tmat)[:3, 3]
                    camera_locations.append(camera_location)
                if success1:

                    # Convert rotation vector to matrix
                    rmat1, _ = cv2.Rodrigues(rvec1)

                    # Create a homogeneous transformation matrix
                    tmat1 = np.hstack((rmat1, tvec1))
                    tmat1 = np.vstack((tmat1, [0, 0, 0, 1]))

                    # Extract camera location (inverse of the transformation matrix)
                    camera_location_1 = np.linalg.inv(tmat1)[:3, 3]
                    camera_locations_1.append(camera_location_1)


        # Convert to numpy array for covariance computation
        camera_locations = np.array(camera_locations)
        camera_locations_1 = np.array(camera_locations_1)
        cov_matrix = np.cov(camera_locations, rowvar=False)
        mean_location = np.mean(camera_locations, axis=0)
    
        # Calculate the error ellipsoid
        X, Y, Z = get_cov_ellipsoid(cov_matrix, mean_location)


        # Plotting the camera locations
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(camera_locations[:, 0], camera_locations[:, 1], camera_locations[:, 2], color='green', s=3, alpha=0.1, label='Noisy')
        ax1.scatter(camera_locations_1[:, 0], camera_locations_1[:, 1], camera_locations_1[:, 2], color='blue', s=10, label='Original')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'Top View - Camera Locations for {numberOfMarkers} Markers')
        ax1.plot_surface(X, Y, Z, alpha=0.2, color='red')
        ax1.view_init(elev=90, azim=-90)
        ax1.set_box_aspect([1, 1, 1])
        ax1.set_xlim(0.108,0.113)
        ax1.set_ylim(-0.150,-0.145)
        ax1.set_zlim(0.135,0.139)
        #ax1.axis('equal')
        
        
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(camera_locations[:, 0], camera_locations[:, 1], camera_locations[:, 2], color='green', s=3, alpha=0.1, label='Noisy')
        ax2.scatter(camera_locations_1[:, 0], camera_locations_1[:, 1], camera_locations_1[:, 2], color='blue', s=10, label='Original')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title(f'Side View - Camera Locations for {numberOfMarkers} Markers')
        ax2.plot_surface(X, Y, Z, alpha=0.2, color='red')
        ax2.view_init(elev=0, azim=0)  
        ax2.set_box_aspect([1, 1, 1])
        ax2.set_xlim(0.108,0.113)
        ax2.set_ylim(-0.150,-0.145)
        ax2.set_zlim(0.135,0.139)
        #ax2.axis('equal')
        plt.show()
    
        # Clear the stack for the next iteration
        camera_locations = []
        camera_locations_1 = []


if __name__ == '__main__':
    montecarlo_simulation()

    