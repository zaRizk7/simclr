import torch
import torch.nn.functional as F

__all__ = ["SimCLRLoss"]


class SimCLRLoss(torch.nn.Module):
    def __init__(self, tau=0.1, reduction="mean"):
        self.tau = tau
        self.reduction = reduction
        super().__init__()

    def forward(self, outputs_1, outputs_2):
        sim = _pairwise_similarity(outputs_1, outputs_2)
        sim = sim / self.tau
        sim = sim.exp()

        loss = 0
        batch_size = sim.size(0)
        mask = torch.arange(batch_size).to(sim.device)
        for i in range(0, batch_size, 2):
            j = i + 1

            m1 = mask != i
            m2 = mask != j
            a, b = sim[i], sim[j]

            loss = (
                loss
                + a[m1].sum().log()
                - a[j].log()
                + b[m2].sum().log()
                - b[i].sum().log()
            )

        return loss / batch_size


def _mask(outputs):
    batch_size = outputs.size(0)
    return ~torch.eye(batch_size).bool().to(outputs.device)


def _norm(outputs):
    return outputs.norm(p=2, dim=-1)


def _pairwise_similarity(outputs_1, outputs_2):
    batch_size = outputs_1.size(0)
    outputs = torch.cat((outputs_1.unsqueeze(1), outputs_2.unsqueeze(1)), dim=1)
    outputs = outputs.view(2 * batch_size, -1)

    norm = _norm(outputs)
    sim = outputs @ outputs.t()
    sim = sim / torch.outer(norm, norm)

    return sim


def nt_xent_loss(outputs_1, outputs_2, tau=0.1):
    batch_size = outputs_1.size(0)
    outputs = torch.cat((outputs_1.unsqueeze(1), outputs_2.unsqueeze(1)), dim=1)
    outputs = outputs.view(2 * batch_size, -1)

    norm = _norm(outputs)
    sim = outputs @ outputs.t()
    sim = sim / torch.outer(norm, norm)

    loss = sim / tau
    loss = loss.exp()

    mask = _mask(loss)
    loss = (loss * mask).sum(-1).log() - loss.diag().log()

    return loss
