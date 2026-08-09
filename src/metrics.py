import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-5):
        """
        Combines Binary Cross Entropy and Dice Loss.

        Args:
            bce_weight (float): Weight for the BCE loss component.
            dice_weight (float): Weight for the Dice loss component.
            smooth (float): A small constant to prevent division by zero.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        # 1. Compute BCE Loss (uses logits directly for numerical stability)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)

        # 2. Compute Dice Loss
        # Apply sigmoid to logits to get probabilities (values between 0 and 1)
        probs = torch.sigmoid(logits)

        # Flatten tensors to compute overlap over the entire batch
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        # 3. Combine the losses
        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)
