"""
Step 5: Train the CNN on the brain scan data.

Handles class imbalance using class weighting, so rare classes
(like Moderate Dementia, 488 images) aren't ignored in favor of
common classes (like Non Demented, 67,222 images).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from dataloaders_by_patient import build_dataloaders_by_patient
from model import BrainScanCNN

# ------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------
N_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available()
                       else "cpu")

print(f"Using device: {DEVICE}")


def compute_class_weights(full_dataset):
    labels = [label for _, label in full_dataset.samples]
    class_counts = np.bincount(labels)

    print(f"Class counts: {dict(zip(full_dataset.idx_to_class.values(), class_counts))}")

    total = len(labels)
    weights = total / (len(class_counts) * class_counts)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    print(f"Class weights: {weights_tensor}")

    return weights_tensor


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    train_loader, val_loader, test_loader, full_dataset = build_dataloaders_by_patient()

    class_weights = compute_class_weights(full_dataset).to(DEVICE)

    model = BrainScanCNN(n_classes=4).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(N_EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{N_EPOCHS} =====")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  New best model saved! (Val Acc: {val_acc:.4f})")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
