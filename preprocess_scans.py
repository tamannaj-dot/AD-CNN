"""
Preprocessing pipeline for brain scan images (PNG/JPG) for AD diagnosis project.

Your raw_scans folder is organized like this:
raw_scans/
    Mild Dementia/
        image1.png
        image2.png
    Non Demented/
        image1.png
        ...
    Moderate Dementia/
        ...
    Very mild Dementia/
        ...

This script goes into EACH subfolder, processes every image inside,
and remembers which subfolder (label) each image came from.

Steps:
1. Load images
2. Convert to grayscale (single channel)
3. Resize to a consistent target size
4. Standardize pixel values (zero mean, unit variance)
"""

import os
import numpy as np
from PIL import Image

# ------------------------------------------------------------------
# STEP 0: SETTINGS - change these to match your project
# ------------------------------------------------------------------
INPUT_DIR = "raw_scans"          # folder containing the 4 class subfolders
OUTPUT_DIR = "processed_scans"   # where cleaned images will be saved
TARGET_SIZE = (224, 224)          # (width, height) - common choice for CNNs

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_image(filepath):
    """STEP 1: Load an image file and return it as a PIL Image object."""
    img = Image.open(filepath)
    return img


def to_grayscale(img):
    """STEP 2: Convert to single-channel grayscale.
    Brain scans should have one intensity value per pixel, not RGB color.
    'L' mode = 8-bit grayscale (0-255).
    """
    return img.convert("L")


def resize_image(img, target_size=TARGET_SIZE):
    """STEP 3: Resize to a consistent size so every image matches.
    LANCZOS is a high-quality resizing method (good for medical images,
    avoids introducing blocky artifacts).
    """
    return img.resize(target_size, Image.LANCZOS)


def standardize_pixels(img):
    """STEP 4: Standardize pixel intensities.

    First converts the image to a numpy array of numbers (0-255),
    then rescales to 0-1, then standardizes to mean=0, std=1.

    Returns a numpy array, ready for a neural network.
    """
    arr = np.array(img, dtype=np.float32)  # shape: (height, width)

    # Normalize to 0-1 range first
    arr = arr / 255.0

    # Standardize: subtract mean, divide by standard deviation
    mean = arr.mean()
    std = arr.std()
    if std > 0:  # avoid dividing by zero on a blank image
        arr = (arr - mean) / std
    else:
        arr = arr - mean

    return arr


def preprocess_one_image(filepath):
    """Run all 4 steps on a single image file."""
    img = load_image(filepath)
    img = to_grayscale(img)
    img = resize_image(img)
    arr = standardize_pixels(img)
    return arr


def preprocess_folder(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    """
    Run the full pipeline on every image, walking through each
    class subfolder inside input_dir.

    Saves:
      - one .npy file per processed image
      - _all_images_stacked.npy  (all images as one big array)
      - _all_labels.npy          (matching label for each image, as text)
    """
    valid_extensions = (".png", ".jpg", ".jpeg")

    # Find the class subfolders (e.g. "Mild Dementia", "Non Demented", ...)
    class_names = [d for d in os.listdir(input_dir)
                   if os.path.isdir(os.path.join(input_dir, d))]

    print(f"Found {len(class_names)} class folders: {class_names}")

    processed_arrays = []
    labels = []

    for class_name in class_names:
        class_folder = os.path.join(input_dir, class_name)
        filenames = [f for f in os.listdir(class_folder)
                     if f.lower().endswith(valid_extensions)]

        print(f"\nProcessing '{class_name}': {len(filenames)} image(s)")

        # Make a matching output subfolder so processed files stay organized
        out_class_folder = os.path.join(output_dir, class_name)
        os.makedirs(out_class_folder, exist_ok=True)

        for fname in filenames:
            filepath = os.path.join(class_folder, fname)
            try:
                arr = preprocess_one_image(filepath)
                processed_arrays.append(arr)
                labels.append(class_name)

                out_name = os.path.splitext(fname)[0] + ".npy"
                np.save(os.path.join(out_class_folder, out_name), arr)

            except Exception as e:
                print(f"  FAILED on {fname}: {e}")

        print(f"  Done with '{class_name}'")

    if processed_arrays:
        stacked = np.stack(processed_arrays)   # shape: (n_images, height, width)
        labels_arr = np.array(labels)            # shape: (n_images,)

        np.save(os.path.join(output_dir, "_all_images_stacked.npy"), stacked)
        np.save(os.path.join(output_dir, "_all_labels.npy"), labels_arr)

        print(f"\nAll done!")
        print(f"Total images processed: {len(processed_arrays)}")
        print(f"Stacked array shape: {stacked.shape}")
        print(f"Labels array shape: {labels_arr.shape}")
        print(f"Saved to: {output_dir}")

    return processed_arrays, labels


if __name__ == "__main__":
    preprocess_folder()


import os 
import numpy as np 
from PIL import Image

#Step 0
INPUT_DIR = "raw_scans"
OUTPUT_DIR = "processed_scans"
TARGET_SIZE = (224,224)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_image(filepath):
    img = Image.open(filepath)
    return img

def to_grayscale(img):
    return img.convert("L")

def resize_image(img, target_size = TARGET_SIZE):
    return img.resize(target_size, Image.LANCZOS)

def standardized_pixels(img):
    arr = np.array(img, dtype = np.float32)

    arr = arr/255.0

    mean = arr.mean()
    std = arr.std()

    if std > 0:
        arr = (arr-mean)/std
    else:
        arr = arr - mean

    return arr

def preprocess_one_image(filepath):
    img = load_image(filepath)
    img = to_grayscale(img)
    img = resize_image(img)
    arr = standardized_pixels(img)

    return arr 

def preprocess_folder(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    valid_extensions = (".png", ".jpg", ".jpeg")
    filenames = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    print(f"Found {len(filenames)} image(s) to process in {input_dir}")

    processed_arrays = []
    for fname in filenames:
        filepath = os.path.join(input_dir, fname)
        try:
            arr = preprocess_one_image(filepath)
            processed_arrays.append(arr)

            # Save each processed array individually
            out_name = os.path.splitext(fname)[0] + ".npy"
            np.save(os.path.join(output_dir, out_name), arr)

            print(f"  Processed: {fname} -> shape {arr.shape}, "
                  f"mean={arr.mean():.3f}, std={arr.std():.3f}")
        except Exception as e:
            print(f"  FAILED on {fname}: {e}")

    if processed_arrays:
        stacked = np.stack(processed_arrays)  # shape: (n_images, height, width)
        np.save(os.path.join(output_dir, "_all_images_stacked.npy"), stacked)
        print(f"\nDone! Stacked array shape: {stacked.shape}")
        print(f"Saved individual + stacked arrays to: {output_dir}")

    return processed_arrays


if __name__ == "__main__":
    preprocess_folder()
