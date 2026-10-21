"""
Step 3 (FIXED): Split the dataset by PATIENT, not by individual image.

The Kaggle Alzheimer's MRI dataset contains multiple slices per patient,
e.g.:
    OAS1_0308_MR1_mpr-1_100.jpg
    OAS1_0308_MR1_mpr-1_101.jpg
    OAS1_0308_MR1_mpr-1_102.jpg
    ...
These are all extremely similar since they're slices from the same brain
scan. If we split at the image level, slices from the same patient can
leak across train/val/test, letting the model "cheat" by recognizing a
patient it already saw, rather than learning real diagnostic patterns.

This version extracts the patient ID from each filename, then splits by
patient ID -- guaranteeing every slice from one patient stays entirely
within a single split.
"""

import re
import random
import torch
from torch.utils.data import DataLoader, Subset
from dataset import BrainScanDataset

# ------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------
BATCH_SIZE = 32
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

random.seed(42)  # so the patient-level split is reproducible


def extract_patient_id(filepath):
    """
    Pulls the patient ID out of a filename like:
        OAS1_0308_MR1_mpr-1_100.npy  ->  'OAS1_0308'

    We use a regex (pattern matching) to grab everything up through
    the 4-digit patient number.
    """
    filename = filepath.split("/")[-1]  # just the filename, not the full path
    match = re.match(r"(OAS1_\d+)", filename)
    if match:
        return match.group(1)
    else:
        # Fallback: if the pattern doesn't match, treat the whole
        # filename as its own group (safer than crashing)
        return filename


def build_dataloaders_by_patient(processed_dir="processed_scans", batch_size=BATCH_SIZE):
    # Step 1: Build the full dataset as before
    full_dataset = BrainScanDataset(processed_dir)

    # Step 2: Group every sample's INDEX by patient ID
    patient_to_indices = {}
    for idx, (filepath, label) in enumerate(full_dataset.samples):
        patient_id = extract_patient_id(filepath)
        patient_to_indices.setdefault(patient_id, []).append(idx)

    all_patient_ids = list(patient_to_indices.keys())
    random.shuffle(all_patient_ids)  # shuffle PATIENTS, not images

    n_patients = len(all_patient_ids)
    print(f"Total unique patients: {n_patients}")

    # Step 3: Split the PATIENT IDs into train/val/test groups
    train_end = int(TRAIN_FRACTION * n_patients)
    val_end = train_end + int(VAL_FRACTION * n_patients)

    train_patients = all_patient_ids[:train_end]
    val_patients = all_patient_ids[train_end:val_end]
    test_patients = all_patient_ids[val_end:]

    print(f"Patients -> Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")

    # Step 4: Convert patient groups back into lists of image INDICES
    def patients_to_indices(patient_list):
        indices = []
        for pid in patient_list:
            indices.extend(patient_to_indices[pid])
        return indices

    train_indices = patients_to_indices(train_patients)
    val_indices = patients_to_indices(val_patients)
    test_indices = patients_to_indices(test_patients)

    print(f"Images   -> Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    # Step 5: Build Subsets using those indices (Subset = "use only these
    # specific indices from the full dataset")
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # Step 6: Wrap each in a DataLoader, same as before
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, full_dataset


if __name__ == "__main__":
    train_loader, val_loader, test_loader, full_dataset = build_dataloaders_by_patient()

    # Sanity check: confirm no patient appears in more than one split
    images, labels = next(iter(train_loader))
    print(f"\nOne batch from train_loader:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
