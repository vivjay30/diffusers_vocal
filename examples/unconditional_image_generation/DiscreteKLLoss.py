import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscreteKLLoss(nn.Module):
    def __init__(self, num_bins=50, min_val=None, max_val=None, epsilon=1e-8):
        super().__init__()
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val
        self.epsilon = epsilon

    def forward(self, pred, target):
        # Determine range for binning
        if self.min_val is None or self.max_val is None:
            with torch.no_grad():
                combined = torch.cat([pred, target])
                self.min_val = combined.min().item()
                self.max_val = combined.max().item()

        # Create bin edges
        bin_edges = torch.linspace(self.min_val, self.max_val, self.num_bins + 1, device=pred.device)
        bin_widths = bin_edges[1:] - bin_edges[:-1]

        # Compute soft histogram
        def soft_histogram(x):
            x_expanded = x.unsqueeze(-1)
            deltas = torch.abs(x_expanded - bin_edges[:-1].unsqueeze(0))
            weights = torch.clamp(1 - deltas / bin_widths, min=0, max=1)
            hist = weights.sum(dim=0) / len(x)
            return hist

        pred_hist = soft_histogram(pred)
        target_hist = soft_histogram(target)

        # Add epsilon and normalize
        pred_probs = (pred_hist + self.epsilon) / (pred_hist.sum() + self.num_bins * self.epsilon)
        target_probs = (target_hist + self.epsilon) / (target_hist.sum() + self.num_bins * self.epsilon)

        # Compute KL divergence
        kl_div = F.kl_div(pred_probs.log(), target_probs, reduction='sum')

        return kl_div
