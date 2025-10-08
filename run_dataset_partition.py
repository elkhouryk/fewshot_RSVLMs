import os
import shutil
from collections import defaultdict
from math import ceil
import argparse
import random

# Set up argument parser
parser = argparse.ArgumentParser(description="Split dataset into train, validation, and test sets.")
parser.add_argument('--dataset_name', type=str, choices=['aid', 'eurosat', 'mlrsnet', 'optimal31', 'patternnet', 'resisc45', 'rsc11', 'rsicb128', 'rsicb256','whurs19'], help='Name of the dataset')

# Parse arguments
args = parser.parse_args()
dataset_name = args.dataset_name

# Define base path and dataset paths
base_path = f"./datasets/{dataset_name}/images/"
train_path = os.path.join(base_path, "train")
val_path = os.path.join(base_path, "val")
test_path = os.path.join(base_path, "test")

# Ensure the images directory exists
if not os.path.exists(base_path):
    raise FileNotFoundError(f"The directory '{base_path}' does not exist. Please provide a valid dataset name.")

# Dictionary to store file paths for each class
class_files = defaultdict(list)

# Iterate through files to categorize by class
for filename in os.listdir(base_path):
    file_path = os.path.join(base_path, filename)
    if os.path.isfile(file_path) and "_" in filename:  # Ensure the filename matches the expected structure
        class_name = filename.split("_")[0]
        class_files[class_name].append(filename)

# Check if there are any valid image files
if not class_files:
    raise ValueError(f"No image files found in the directory '{base_path}'. Ensure the directory contains files with the format '{{classname}}_{{id}}'.")

# Create train, validation, and test directories if they don't exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Set a fixed random seed for reproducibility
random.seed(42)

# Split files into train, validation, and test sets
for class_name, files in class_files.items():
    # Shuffle files for randomness
    random.shuffle(files)
    
    # Calculate split indices
    total_files = len(files)
    train_idx = ceil(total_files * 0.50)
    val_idx = train_idx + ceil(total_files * 0.25)
    
    # Split files
    train_files = files[:train_idx]
    val_files = files[train_idx:val_idx]
    test_files = files[val_idx:]
    
    # Move train files
    for file in train_files:
        src = os.path.join(base_path, file)
        dst = os.path.join(train_path, file)
        shutil.move(src, dst)
    
    # Move validation files
    for file in val_files:
        src = os.path.join(base_path, file)
        dst = os.path.join(val_path, file)
        shutil.move(src, dst)
    
    # Move test files
    for file in test_files:
        src = os.path.join(base_path, file)
        dst = os.path.join(test_path, file)
        shutil.move(src, dst)

print("Files successfully split into train, validation, and test directories!")