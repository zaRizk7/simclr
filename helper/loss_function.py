import torch
import torch.nn.functional as F

__all__ = ["NTXentLoss"]


class NTXentLoss(torch.nn.Module):
    def __init__(self, tau=0.1, reduction="mean"):
        self.tau = tau
        self.reduction = reduction
        super().__init__()

    def forward(self, outputs_1, outputs_2):
        loss = nt_xent_loss(outputs_1, outputs_2, self.tau)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def _pairwise_similarity(outputs_1, outputs_2):
    size = outputs_1.size()
    projection_size = size[-1]

    outputs = torch.stack((outputs_1, outputs_2), 1)
    outputs = outputs.view(-1, projection_size)
    norms = outputs.norm(p=2, dim=tuple(range(1, len(size))))

    similarities = outputs @ outputs.t()
    similarities = similarities / torch.outer(norms, norms)

    return similarities


def nt_xent_loss(outputs_1, outputs_2, tau=0.1):
    similarities = _pairwise_similarity(outputs_1, outputs_2)

    batch_size = similarities.size(0)
    loss = similarities / tau
    loss = loss.exp()

    i = torch.arange(batch_size) % 2 == 0
    j = ~i

    
    masks_1 = torch.zeros_like(loss).bool()
    masks_1[i, j] = 1
    masks_1[j, i] = 1
    masks_2 = ~torch.eye(batch_size, device=masks_1.device).bool()

    loss = (loss * masks_2).sum(-1).log() - (loss * masks_1).sum(-1).log()

    return loss
