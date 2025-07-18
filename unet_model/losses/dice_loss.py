import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        Dice Loss for binary segmentation.
        Args:
            smooth (float): Smoothing factor to prevent division by zero.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        Compute Dice Loss.
        Args:
            preds: Raw logits from the model (before activation), shape (batch_size, 1, H, W)
            targets: Ground truth binary mask, shape (batch_size, 1, H, W)
        Returns:
            Scalar Dice Loss.
        """
        # Convert logits to probabilities
        preds = torch.sigmoid(preds)
        # Ensure targets are floats
        targets = targets.float()

        # Check that shapes match
        if preds.shape != targets.shape:
            raise ValueError(f"Shape mismatch: preds shape {preds.shape} and targets shape {targets.shape} must be identical.")

        # Flatten tensors
        preds_flat = preds.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)

        # Compute Dice score
        intersection = (preds_flat * targets_flat).sum()
        union = preds_flat.sum() + targets_flat.sum()
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice_score  # Dice loss

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1.0, bce_weight=1.0, smooth=1e-6):
        """
        Combines Dice Loss and BCEWithLogitsLoss.
        Args:
            dice_weight (float): Weight for the Dice loss term.
            bce_weight (float): Weight for the BCE loss term.
            smooth (float): Smoothing factor for Dice loss.
        """
        super().__init__()
        self.dice_loss = DiceLoss(smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()  # works directly on logits
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, preds, targets):
        """
        Compute the combined loss.
        Args:
            preds: Raw logits from the model, shape (batch_size, 1, H, W)
            targets: Ground truth binary mask, shape (batch_size, 1, H, W)
        Returns:
            Scalar combined loss.
        """
        dice = self.dice_loss(preds, targets)
        bce = self.bce_loss(preds, targets)
        return self.dice_weight * dice + self.bce_weight * bce


class CustomBCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        Custom implementation of BCEWithLogitsLoss.
        Args:
            reduction (str): Specifies the reduction to apply to the output:
                             'mean' | 'sum' | 'none'. Default: 'mean'
        """
        super().__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError("Reduction must be one of: 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits (Tensor): Raw model outputs (logits), any shape.
            targets (Tensor): Binary targets (0 or 1), same shape as logits.
        Returns:
            Tensor: The computed loss.
        """
        # Ensure the targets are floats for the multiplication
        targets = targets.float()
        
        # Compute the loss in a numerically stable way:
        # loss = max(logits, 0) - logits * targets + log(1 + exp(-|logits|))
        loss = torch.clamp(logits, min=0) - logits * targets + torch.log(1 + torch.exp(-torch.abs(logits)))
        
        # Apply the reduction method
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
