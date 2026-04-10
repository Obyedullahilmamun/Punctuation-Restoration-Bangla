import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter


class FocalLoss(nn.Module):
    """
    Focal Loss for highly imbalanced punctuation classification.
    
    Dynamically scales the standard cross-entropy loss so that well-classified
    (easy) examples contribute less to the total loss, focusing training on
    hard-to-classify minority classes (e.g., exclamation marks).
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    
    Args:
        weight (Tensor, optional): Manual class weights. Default: None.
        gamma (float): Focusing parameter. Higher gamma = stronger focus on
                       hard examples. Default: 2.0.
        ignore_index (int): Target value to ignore in loss computation.
                            Default: -100.
    """
    def __init__(self, weight=None, gamma=2.0, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.weight,
            reduction='none',
            ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # Mask out ignore_index padding tokens
        valid_mask = targets != self.ignore_index
        if valid_mask.sum() > 0:
            return focal_loss[valid_mask].mean()
        else:
            return focal_loss.sum()  # fallback for empty batches


def compute_class_weights(dataset, num_classes=5, device='cpu'):
    """
    Compute inverse-frequency class weights from the training dataset labels.
    
    Scans all label sequences in the dataset and computes:
        weight[c] = total_valid_tokens / (num_classes * count[c])
    
    This ensures that rare classes (e.g., exclamation marks) receive
    proportionally higher weight during loss computation.
    
    Args:
        dataset: Dataset object with .data attribute containing
                 [tokens, labels, attn_mask, label_mask] per sample.
        num_classes (int): Number of punctuation classes. Default: 5.
        device (str): Device to place the weight tensor on.
    
    Returns:
        Tensor of shape (num_classes,) with normalized class weights.
    """
    label_counts = Counter()

    for sample in dataset.data:
        labels = sample[1]       # label sequence
        label_mask = sample[3]   # valid position mask

        for label, mask in zip(labels, label_mask):
            if mask == 1:
                label_counts[label] += 1

    total = sum(label_counts.values())
    weights = torch.zeros(num_classes)

    for c in range(num_classes):
        count = label_counts.get(c, 1)  # avoid division by zero
        weights[c] = total / (num_classes * count)

    # Log the computed weights for reproducibility
    print(f"Class distribution: {dict(sorted(label_counts.items()))}")
    print(f"Computed class weights: {weights.tolist()}")

    return weights.to(device)
