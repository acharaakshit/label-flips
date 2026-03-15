from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, include_background: bool = True, smooth: float = 1e-5):
        super().__init__()
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_onehot = F.one_hot(targets.long(), num_classes).permute(0, 3, 1, 2).float()
        start_class = 0 if self.include_background else 1
        dice_scores = []
        for c in range(start_class, num_classes):
            p = probs[:, c]
            t = targets_onehot[:, c]
            inter = (p * t).sum(dim=(1, 2))
            denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
            dice = (2 * inter + self.smooth) / (denom + self.smooth)
            dice_scores.append(dice)
        loss = 1.0 - torch.stack(dice_scores, dim=1).mean()
        return loss


class DiceCELoss(nn.Module):
    def __init__(self, include_background: bool = True, smooth: float = 1e-5):
        super().__init__()
        self.dice = DiceLoss(include_background=include_background, smooth=smooth)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets) + self.dice(logits, targets)


class GroupDROLoss(nn.Module):
    def __init__(self, n_groups: int = 4, group_weight_step: float = 0.01, adjustment: float = 0.0):
        super().__init__()
        self.n_groups = n_groups
        self.group_weight_step = group_weight_step
        self.adjustment = adjustment
        self.register_buffer("group_weights", torch.ones(n_groups) / n_groups)
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        per_pixel_loss = self.ce_loss(logits, targets)
        per_sample_loss = per_pixel_loss.mean(dim=(1, 2))

        device = logits.device
        group_ids = group_ids.to(device)

        group_losses = torch.zeros(self.n_groups, device=device)
        group_counts = torch.zeros(self.n_groups, device=device)
        for g in range(self.n_groups):
            mask = group_ids == g
            if mask.any():
                group_losses[g] = per_sample_loss[mask].mean()
                group_counts[g] = mask.sum()

        valid = group_counts > 0
        if not valid.any():
            return per_sample_loss.mean(), {"note": "no valid groups in batch"}

        adjusted_losses = group_losses + (
            self.adjustment
            if torch.is_tensor(self.adjustment)
            else torch.tensor(self.adjustment, device=device)
        )

        with torch.no_grad():
            new_w = self.group_weights.clone()
            new_w[valid] = new_w[valid] * torch.exp(self.group_weight_step * adjusted_losses[valid])
            new_w[~valid] = 0
            new_w = new_w / new_w.sum()
            self.group_weights.copy_(new_w)

        weighted_loss = (self.group_weights[valid] * group_losses[valid]).sum()

        metrics = {}
        for g in range(self.n_groups):
            metrics.update(
                {
                    f"group_{g}_loss": group_losses[g].item(),
                    f"group_{g}_weight": self.group_weights[g].item(),
                    f"group_{g}_count": group_counts[g].item(),
                }
            )

        return weighted_loss, metrics
