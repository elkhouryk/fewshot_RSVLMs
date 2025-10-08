import os
import shutil
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset split, copy files, and clean inputs.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="PatternNet",
        help='Name of the dataset (default: "PatternNet")'
    )
    return parser.parse_args()

# Function to read class names from the classes.txt file
def read_classes(classes_file):
    try:
        with open(classes_file, "r") as f:
            classes = [line.strip() for line in f.readlines()]
            print(f"Classes read: {classes}")
            return classes
    except Exception as e:
        print(f"Error reading classes.txt: {e}")
        return []

# Function to read human-readable class names from class_changes.txt
def read_class_changes(class_changes_file):
    try:
        with open(class_changes_file, "r") as f:
            class_changes = [line.strip() for line in f.readlines()]
            print(f"Class changes read: {class_changes}")
            return class_changes
    except Exception as e:
        print(f"Error reading class_changes.txt: {e}")
        return []

# Function to generate the JSON structure
def generate_json_structure(images_path, classes, class_changes):
    dataset_split = {"train": [], "val": [], "test": []}  # Added "val" partition

    for partition in ["train", "val", "test"]:  # Include "val" in partitions
        partition_path = os.path.join(images_path, partition)
        print(f"Processing partition: {partition}, path: {partition_path}")
        
        if not os.path.exists(partition_path):
            print(f"Partition path does not exist: {partition_path}")
            continue

        for image_file in os.listdir(partition_path):
            # Skip non-image files
            if not image_file.endswith((".jpg", ".png", ".bmp", ".tif", "tiff")):
                print(f"Skipping non-image file: {image_file}")
                continue

            # Infer the class name from the file name
            class_name = image_file.split("_")[0]  # Extract class name before the "_"
            if class_name not in classes:
                print(f"Class name '{class_name}' not found in classes.txt")
                continue

            # Map class name to index and human-readable name
            label_index = classes.index(class_name)
            human_readable_class = class_changes[label_index]
            
            # Add the image entry to the JSON structure
            image_path = os.path.join(class_name, image_file)
            dataset_split[partition].append([image_path, label_index, human_readable_class])

    return dataset_split

# Function to copy dataset to the output path with the new structure
def copy_dataset_with_class_folders(images_path, classes, output_path):
    os.makedirs(output_path, exist_ok=True)
    for class_name in classes:
        class_output_folder = os.path.join(output_path, class_name)
        os.makedirs(class_output_folder, exist_ok=True)
        print(f"Created output folder for class: {class_output_folder}")
        
        for partition in ["train", "val", "test"]:  # Include "val" in partitions
            partition_path = os.path.join(images_path, partition)
            if not os.path.exists(partition_path):
                print(f"Partition path does not exist: {partition_path}")
                continue

            for image_file in os.listdir(partition_path):
                # Check if the image belongs to the current class
                if not image_file.startswith(class_name + "_"):
                    continue

                src = os.path.join(partition_path, image_file)
                dst = os.path.join(class_output_folder, image_file)
                try:
                    shutil.copy(src, dst)
                    print(f"Copied {src} to {dst}")
                except Exception as e:
                    print(f"Error copying {src} to {dst}: {e}")

def remove_inputs_unconditional(images_path, classes_file, class_changes_file):
    # Delete directory
    if os.path.isdir(images_path):
        try:
            shutil.rmtree(images_path)
            print(f"Removed directory: {images_path}")
        except Exception as e:
            print(f"Error removing directory {images_path}: {e}")
    else:
        print(f"Directory not found (skip): {images_path}")

    # Delete files
    for path in [classes_file, class_changes_file]:
        if os.path.isfile(path):
            try:
                os.remove(path)
                print(f"Removed file: {path}")
            except Exception as e:
                print(f"Error removing file {path}: {e}")
        else:
            print(f"File not found (skip): {path}")

if __name__ == "__main__":
    args = parse_args()

    # Single root: ./datasets/[dataset_name]
    dataset_root = os.path.join(".", "datasets", args.dataset_name)

    # Input paths
    dataset_path = dataset_root
    images_path = os.path.join(dataset_path, "images")
    classes_file = os.path.join(dataset_path, "classes.txt")
    class_changes_file = os.path.join(dataset_path, "class_changes.txt")

    # Output paths (inside the same dataset root)
    output_root = dataset_root
    output_json = os.path.join(output_root, "split.json")
    output_path = os.path.join(output_root, "2750/")

    # Step 1: Read class names
    classes = read_classes(classes_file)
    if not classes:
        print("No classes found. Exiting.")
        exit(1)

    # Step 2: Read human-readable class names
    class_changes = read_class_changes(class_changes_file)
    if not class_changes:
        print("No class changes found. Exiting.")
        exit(1)

    # Step 3: Generate JSON structure
    dataset_split = generate_json_structure(images_path, classes, class_changes)
    print(f"Generated dataset split: {json.dumps(dataset_split, indent=2)}")  # Debugging output

    # Save JSON file in the output path
    os.makedirs(output_root, exist_ok=True)  # Ensure the output directory exists
    try:
        with open(output_json, "w") as f:
            json.dump(dataset_split, f, indent=4)
        print(f"JSON file saved to {output_json}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        exit(1)

    # Step 4: Copy dataset to the new structure
    os.makedirs(output_path, exist_ok=True)
    copy_dataset_with_class_folders(images_path, classes, output_path)
    print(f"Dataset copied to {output_path}")

    # Step 5: Unconditional cleanup of original inputs
    remove_inputs_unconditional(images_path, classes_file, class_changes_file)
    print("Cleanup complete.")