import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torchvision.ops import box_iou
from glob import glob
from shapely.geometry import box
import pandas as pd

NATIVE_IMAGES_DIR = "path/to/image_chips_native/"
#© 2023 Maxar Technologies. 
# This imagery is provided under the Creative Commons Attribution-NonCommercial 4.0 International Public License. 
# (https://creativecommons.org/licenses/by-nc/4.0/legalcode#s3a1Ai)
LABELS_DIR = "path/to/labels_native/"
IMAGE_SIZE = 416  # Native images size = 416x416
RESOLUTION_M = 0.31  # 31 cm resolution

def read_labels(label_path):
    with open(label_path, "r") as f:
        lines = f.readlines()
    return [list(map(float, line.strip().split())) for line in lines]

label_files = glob(os.path.join(LABELS_DIR, "*.txt"))
image_files = glob(os.path.join(NATIVE_IMAGES_DIR, "*.tif"))

num_instances = 0
labels_per_image = []
areas = []

for label_file in label_files:
    labels = read_labels(label_file)
    num_instances += len(labels)
    labels_per_image.append(len(labels))
    
    for label in labels:
        if len(label) < 5:
            continue  # Skip invalid label entries
        _, _, _, width, height = label  # YOLO format (normalized)
        width_m = width * IMAGE_SIZE * RESOLUTION_M
        height_m = height * IMAGE_SIZE * RESOLUTION_M
        areas.append(width_m * height_m)

areas = np.array(areas)

if len(areas) > 0:
    mean_area = np.mean(areas)
    std_area = np.std(areas)
else:
    mean_area = std_area = 0.0

print(f"Total solar panel instances: {num_instances}")
print(f"Mean Area: {mean_area:.4f} m², Std Dev: {std_area:.4f} m²")

if len(areas) > 0:
    plt.hist(areas, bins=30, edgecolor='black')
    plt.xlabel("Solar Panel Area (m²)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Solar Panel Areas")
    plt.show()

if len(labels_per_image) > 0:
    unique, counts = np.unique(labels_per_image, return_counts=True)
    label_count_df = pd.DataFrame({'Labels per Image': unique, 'Image Count': counts})
    print(label_count_df.to_string(index=False))

    plt.figure(figsize=(10, 5))
    plt.bar(unique, counts, color='blue', edgecolor='black')
    plt.xlabel("Number of Labels per Image")
    plt.ylabel("Number of Images")
    plt.title("Distribution of Labels per Image")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
else:
    print("No labels found, skipping label histogram generation.")


print("\nMethod Used for Area Computation:")
print("1. YOLO bounding boxes are normalized (0 to 1 scale).")
print("2. Width and height are multiplied by IMAGE_SIZE to get pixel dimensions.")
print("3. Pixel dimensions are converted to meters using RESOLUTION_M (0.31 m per pixel).")
print("4. Final area is computed as width_m * height_m.")

def compute_iou_torchvision(box1, box2):
    bbox1_tensor = torch.tensor([box1])
    bbox2_tensor = torch.tensor([box2])
    return box_iou(bbox1_tensor, bbox2_tensor)[0, 0].item()

bbox1 = [0.1, 0.1, 0.4, 0.4]  # [x_min, y_min, x_max, y_max]
bbox2 = [0.2, 0.2, 0.5, 0.5]

iou_torchvision = compute_iou_torchvision(bbox1, bbox2)
print(f"IoU (Torchvision): {iou_torchvision:.4f}")

def compute_ap_voc11(recalls, precisions):
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = np.max(precisions[recalls >= t]) if np.any(recalls >= t) else 0
        ap += p / 11.0
    return ap

def compute_ap_coco101(recalls, precisions):
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = np.max(precisions[recalls >= t]) if np.any(recalls >= t) else 0
        ap += p / 101.0
    return ap

# Example for precision-recall curve
recalls = np.linspace(0, 1, 50)
precisions = np.linspace(1, 0.5, 50)

ap_voc11 = compute_ap_voc11(recalls, precisions)
ap_coco101 = compute_ap_coco101(recalls, precisions)

print(f"AP (Pascal VOC 11-point): {ap_voc11:.4f}")
print(f"AP (COCO 101-point): {ap_coco101:.4f}")
