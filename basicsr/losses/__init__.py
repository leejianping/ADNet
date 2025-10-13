"""
Initialization module for losses.
"""
from .losses import (L1Loss, MSELoss, PSNRLoss, CharbonnierLoss,SobelEdgeLoss,BinaryLoss ,QRLoss)

__all__ = [
    'L1Loss', 'MSELoss', 'PSNRLoss', 'CharbonnierLoss',
 'SobelEdgeLoss', 'BinaryLoss', 'QRLoss'
]
