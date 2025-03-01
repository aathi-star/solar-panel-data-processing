Overview
This script processes solar panel image datasets, extracts bounding box annotations from YOLO label files, and computes various statistics such as object count, area distribution, and label frequency. It also includes visualization features for better data understanding.

Features
- Reads YOLO-formatted bounding box labels.
- Computes solar panel instance count.
- Calculates mean and standard deviation of bounding box areas.
- Generates histograms for area distribution and labels per image.
- Implements IoU (Intersection over Union) computation using `torchvision`.
- Calculates Average Precision (AP) using Pascal VOC 11-point and COCO 101-point interpolation methods.

Requirements
Make sure the following dependencies are installed before running the script:

pip install numpy matplotlib opencv-python torch torchvision shapely pandas

Usage
Update the following paths in the script to point to the correct dataset locations:

NATIVE_IMAGES_DIR = "path/to/image_chips_native/"
LABELS_DIR = "path/to/labels_native/"


Then, run the script using:

python processingscript.py

Dataset Information
- **Image Size**: 416x416 pixels
- **Resolution**: 0.31 meters per pixel
- **Bounding Box Format**: YOLO (normalized values between 0 and 1)

Output
- Total count of detected solar panels
- Mean and standard deviation of solar panel areas (in square meters)
- Histogram plots for:
  - Solar panel area distribution
  - Labels per image distribution
- Validation of image loading
- Computed IoU values for sample bounding boxes
- Computed AP values using Pascal VOC 11-point and COCO 101-point methods

License
This dataset contains imagery from Maxar Technologies and is provided under the [Creative Commons Attribution-NonCommercial 4.0 International Public License](https://creativecommons.org/licenses/by-nc/4.0/legalcode#s3a1Ai).

Author
Aathithya Sharan A

