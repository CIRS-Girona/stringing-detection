# Stringing Detection

This repository contains the python scripts for replicating the approach presented in "*Real-time Stringing Detection for Additive Manufacturing*" published in the *Journal of Manufacturing and Materials Processing, February 2025, DOI: https://doi.org/10.3390/jmmp9030074*

# Computer vision approach

## Prerequisites
The file ```requirements.txt``` contains the necessary Python packages for this project. To install, run:

    pip install -r requirements.txt

## Usage

The ```source``` directory contains python scripts implementing the computer vision approach for detecting stringing defects in 3D printing.

```
    ├── collect_data.py                 # Script to collect images of printed objects.
    ├── compare_images.py               # Compares synthetic and printed object images.
    ├── monocular_calibration.py        # Camera calibration for accurate 3D-2D mapping.
    ├── montecarlo_simulation.py        # Simulates detection for robustness testing.
    ├── morphological_operation.py      # Refines images using morphological operations.
    ├── segmentation.py                 # Segments images based on color.
    └── solve_pnp.py                    # Solves Perspective-n-Point for object localization.
```

Below is an example of how to run the data collection script. The other scripts can be run in a similar way, with parameters adjusted based on your hardware setup. For detailed information about each script’s parameters, use the --help flag.

```
python .\collect_data.py --serial_port COM4 --baud_rate 250000 --microscope_index 0 --cam_index 1 --micro_dir   micro_folder --flir_dir flir_folder
```

# Machine learning approach
The code for training and evaluation is not included in this repository, as the necessary scripts are directly provided by the Darknet framework. Users are encouraged to refer to the [repository](https://github.com/AlexeyAB/darknet) maintained by AlexeyAB for a step-by-step guide.

However, the pretrained model that achieves the best results, as reported in the paper, along with the corresponding configuration file, can be found at [this](https://drive.google.com/drive/folders/1d7YOXVxjQ_8nzqWe7IAnf6rIWBKOBMHv) link.

# Citation
If you find this repository, please consider giving us a star :star:

```
@Article{jmmp9030074,
    AUTHOR = {Charia, Oumaima and Rajani, Hayat and Ferrer Real, Inés and Domingo-Espin, Miquel and Gracias, Nuno},
    TITLE = {Real-Time Stringing Detection for Additive Manufacturing},
    JOURNAL = {Journal of Manufacturing and Materials Processing},
    VOLUME = {9},
    YEAR = {2025},
    NUMBER = {3},
    ARTICLE-NUMBER = {74},
    URL = {https://www.mdpi.com/2504-4494/9/3/74},
    ISSN = {2504-4494},
    DOI = {10.3390/jmmp9030074}
}
```