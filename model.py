"""
Step 4: Define the CNN model architecture.

Input:  a batch of grayscale brain scan images, shape (batch, 1, 224, 224)
Output: a batch of predictions across 4 classes
        (Mild Dementia, Moderate Dementia, Non Demented, Very mild Dementia)
"""

import torch
import torch.nn as nn


class BrainScanCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()  # required setup for any nn.Module

        # ------------------------------------------------------------
        # PART 1: Convolutional "feature detector" layers
        # ------------------------------------------------------------
        # Each block does: slide filters -> zero out negatives -> shrink

        # Block 1: look for simple patterns (edges, basic textures)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # 224x224 -> 112x112

        # Block 2: combine simple patterns into more complex ones
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # 112x112 -> 56x56

        # Block 3: even more complex/abstract patterns
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)  # 56x56 -> 28x28

        # ------------------------------------------------------------
        # PART 2: Fully-connected "decision" layers
        # ------------------------------------------------------------
        self.flatten = nn.Flatten()

        # After 3 pooling steps: 224 -> 112 -> 56 -> 28
        # So each feature map is 28x28, and we have 64 of them (from conv3)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # helps prevent overfitting, explained below

        self.fc2 = nn.Linear(128, n_classes)  # final output: one score per class

    def forward(self, x):
        # Pass through conv block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Pass through conv block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Pass through conv block 3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Flatten from (batch, 64, 28, 28) into (batch, 64*28*28)
        x = self.flatten(x)

        # Decision layers
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)  # raw scores, one per class (not probabilities yet)

        return x


if __name__ == "__main__":
    # Quick sanity check: create the model and run one fake batch through it
    model = BrainScanCNN(n_classes=4)

    # Fake batch: 8 images, 1 channel, 224x224 (just random numbers, for testing shape only)
    fake_batch = torch.randn(8, 1, 224, 224)

    output = model(fake_batch)
    print(f"Input shape:  {fake_batch.shape}")
    print(f"Output shape: {output.shape}")  # should be (8, 4) - 8 images, 4 class scores each

    # Count total learnable parameters, just to see how big this model is
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
