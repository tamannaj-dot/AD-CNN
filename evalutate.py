"""
Step 6: Final evaluation on the TEST set.

This is the one true, honest measure of how well the model performs
on brain scans it has never seen in any form during training or
validation-based decision making.
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from dataloaders_by_patient import build_dataloaders_by_patient
from model import BrainScanCNN

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available()
                       else "cpu")


def evaluate_on_test_set():
    # Step 1: Rebuild the exact same data splits (same random seed = same patients)
    train_loader, val_loader, test_loader, full_dataset = build_dataloaders_by_patient()

    # Step 2: Load the saved best model
    model = BrainScanCNN(n_classes=4).to(DEVICE)
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    model.eval()  # turn off dropout, we're only evaluating now

    all_predictions = []
    all_true_labels = []

    # Step 3: Run every test image through the model, no training happening
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.numpy())

    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)

    # Step 4: Overall test accuracy
    test_acc = (all_predictions == all_true_labels).mean()
    print(f"\n{'='*50}")
    print(f"FINAL TEST ACCURACY: {test_acc:.4f}")
    print(f"{'='*50}\n")

    # Step 5: Confusion matrix -- which classes get mixed up with which
    class_names = [full_dataset.idx_to_class[i] for i in range(4)]

    cm = confusion_matrix(all_true_labels, all_predictions)
    print("Confusion Matrix:")
    print("(rows = true label, columns = predicted label)\n")

    header = " " * 22 + "".join([f"{name[:12]:>14}" for name in class_names])
    print(header)
    for i, row in enumerate(cm):
        row_str = "".join([f"{val:>14}" for val in row])
        print(f"{class_names[i][:20]:>20}{row_str}")

    # Step 6: Detailed per-class breakdown (precision, recall, f1-score)
    print("\nPer-class performance:")
    print(classification_report(all_true_labels, all_predictions, target_names=class_names))

    return test_acc, cm


if __name__ == "__main__":
    evaluate_on_test_set()
