# innovative_models/utils/losses.py
import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    """Pinball loss for quantile regression"""

    def __init__(self, quantiles=[0.05, 0.5, 0.95]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """preds: [batch, num_quantiles]"""
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))

        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss