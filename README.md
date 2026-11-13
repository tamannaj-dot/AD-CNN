Alzheimer's Disease Staging from Brain MRI with a CNN

A PyTorch convolutional neural network that classifies brain MRI slices into four stages of dementia: Non Demented, Very mild Dementia, Mild Dementia, and Moderate Dementia.

Educational/research project. Not a medical device and not for clinical use.

Key points
Patient-level data splitting. The dataset contains many slices per patient. My first version split at the image level, which lets near-identical slices from the same patient land in both train and test (data leakage). dataloaders_by_patient.py groups slices by patient ID so every patient appears in exactly one split. The original naive split is kept in dataloaders_naive_split.py for comparison.
Class imbalance handling. The classes are heavily imbalanced (about 67,000 Non Demented slices vs. 488 Moderate Dementia), so training uses inverse-frequency class weights in the loss.
Custom pipeline. Preprocessing (grayscale, resize to 224x224, per-image standardization), a custom Dataset, training with best-checkpoint selection on validation accuracy, and a held-out test evaluation with a confusion matrix and per-class precision/recall/
