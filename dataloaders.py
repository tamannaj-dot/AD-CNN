"""
Step 3: Split the dataset into train/validation/test sets,
and wrap each one in a DataLoader for batching.

Run this AFTER dataset.py has been created (we import BrainScanDataset from it).
"""

import torch
from torch.utils.data import DataLoader, random_split
from dataset import BrainScanDataset

# ------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------
BATCH_SIZE = 32
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15  # should add up to 1.0 with the two above

# Set a fixed random seed so the split is the same every time you run this.
# Without this, you'd get a DIFFERENT random split each run, which makes
# results hard to compare and reproduce.
torch.manual_seed(42)


def build_dataloaders(processed_dir="processed_scans", batch_size=BATCH_SIZE):
    # Step 1: Build the full dataset (all 86,437 images, as file paths + labels)
    full_dataset = BrainScanDataset(processed_dir)

    total_size = len(full_dataset)
    train_size = int(TRAIN_FRACTION * total_size)
    val_size = int(VAL_FRACTION * total_size)
    test_size = total_size - train_size - val_size  # remainder, avoids rounding gaps

    print(f"\nSplitting {total_size} total images into:")
    print(f"  Train: {train_size}")
    print(f"  Val:   {val_size}")
    print(f"  Test:  {test_size}")

    # Step 2: Randomly split into 3 non-overlapping groups
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # Step 3: Wrap each split in a DataLoader
    # shuffle=True for training: mixes up image order each epoch, so the
    #   network doesn't learn any accidental order-based patterns.
    # shuffle=False for val/test: no need to shuffle since we're just
    #   evaluating, not learning from the order.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, full_dataset


if __name__ == "__main__":
    train_loader, val_loader, test_loader, full_dataset = build_dataloaders()

    # Quick sanity check: grab just ONE batch and look at its shape
    images, labels = next(iter(train_loader))
    print(f"\nOne batch from train_loader:")
    print(f"  Images shape: {images.shape}")  # expect (32, 1, 224, 224)
    print(f"  Labels shape: {labels.shape}")  # expect (32,)
    print(f"  Example labels in this batch: {labels[:10].tolist()}")
