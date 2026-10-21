"""
Custom PyTorch Dataset for the AD diagnosis project.

This class knows how to:
1. Walk through the processed_scans folder (4 class subfolders)
2. Build a master list of every image file + its label
3. Load one specific image on demand (when the DataLoader asks for it)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class BrainScanDataset(Dataset):
    def __init__(self, processed_dir="processed_scans"):
        """
        Runs once, when you first create the dataset.
        Builds a list of (filepath, label) pairs -- does NOT load
        any actual image data yet, just remembers where everything is.
        """
        self.samples = []  # will hold tuples of (filepath, label_name)

        # Find the class subfolders, e.g. "Mild Dementia", "Non Demented", ...
        class_names = sorted([
            d for d in os.listdir(processed_dir)
            if os.path.isdir(os.path.join(processed_dir, d))
        ])

        # Turn class names into numbers, since PyTorch needs numeric labels
        # e.g. {"Mild Dementia": 0, "Moderate Dementia": 1, ...}
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.idx_to_class = {i: name for name, i in self.class_to_idx.items()}

        print(f"Classes found: {self.class_to_idx}")

        for class_name in class_names:
            class_folder = os.path.join(processed_dir, class_name)
            npy_files = [f for f in os.listdir(class_folder) if f.endswith(".npy")]

            for fname in npy_files:
                filepath = os.path.join(class_folder, fname)
                label_idx = self.class_to_idx[class_name]
                self.samples.append((filepath, label_idx))

        print(f"Total samples found: {len(self.samples)}")

    def __len__(self):
        """
        Tells PyTorch how many total images there are.
        Called automatically whenever something needs to know the dataset size.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        The core method: given an index (like 4502), load and return
        that specific image + its label.

        This is called automatically by the DataLoader, repeatedly,
        for whichever indices it needs for the next batch.
        """
        filepath, label = self.samples[idx]

        # Load the standardized pixel array we saved earlier
        arr = np.load(filepath)  # shape: (224, 224)

        # PyTorch expects images as (channels, height, width), not just (height, width)
        # Our images are grayscale, so we add a "1" for a single channel
        arr = np.expand_dims(arr, axis=0)  # shape becomes (1, 224, 224)

        # Convert to a PyTorch tensor (PyTorch's version of a numpy array)
        image_tensor = torch.tensor(arr, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor


if __name__ == "__main__":
    # Quick test: make sure the dataset loads correctly
    dataset = BrainScanDataset("processed_scans")

    # Try grabbing the very first sample to sanity check it works
    image, label = dataset[0]
    print(f"\nFirst sample:")
    print(f"  Image shape: {image.shape}")
    print(f"  Label: {label.item()} ({dataset.idx_to_class[label.item()]})")
