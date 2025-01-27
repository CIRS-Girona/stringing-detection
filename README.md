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
