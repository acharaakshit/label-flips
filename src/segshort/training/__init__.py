"""Training module for segmentation shortcuts research."""

# Baseline losses (with citations)
from .baseline_losses import DiceLoss, DiceCELoss, GroupDROLoss

__all__ = [
    "DiceLoss",
    "DiceCELoss",
    "GroupDROLoss",
]
