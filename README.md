# stringing-detection
This repository contains a python code for detecting stringing in 3D printing, as proposed in "Real-time Stringing Detection for Additive Manufacturing" published in Journal of Manufacturing and Materials Processing, DOI:....... The detection is achieved using both machine learning and computer vision approaches.
# Getting Started
# Computer vision approach
## Prerequisites
The file requirements.txt contains the necessary Python packages for this project. To install, run:

    pip install -r requirements.txt

This repository contains python scripts implementing the computer vision approach for stringing detection in 3D printing.

    ├── collect_data.py                 # Script to collect images of printed objects.
    ├── compare_images.py               # Compares synthetic and printed object images.
    ├── monocular_calibration.py        # Camera calibration for accurate 3D-2D mapping.
    ├── montecarlo_simulation.py        # Simulates detection for robustness testing.
    ├── morphological_operation.py      # Refines images using morphological operations.
    ├── segmentation.py                 # Segments images based on color.
    ├── solve_pnp.py                    # Solves Perspective-n-Point for object localization.

## How to use: 
Below is an example of how to run the first script in this repository. The other scripts can be run in a similar way, with parameters adjusted based on your hardware setup. For detailed information about each script’s parameters, use the --help flag

    python .\collect_data.py --serial_port COM4 --baud_rate 250000 --microscope_index 0 --cam_index 1 --micro_dir   micro_folder --flir_dir flir_folder

# Machine learning approach
We have not included specific code related to this section in the repository because the necessary tools and commands for training and evaluation are provided by the Darknet framework itself.
For a detailed tutorial on how to use Darknet, refer to the following link:

Darknet YOLO Tutorial

Additionally, you can find the pretrained model that achieves high performance as well as the configuration file used in the link below:
