"""
Loss functions for the NLLReg study.

Includes SIGReg (Sketched Isotropic Gaussian Regularization) and its supervised
class-conditional variants, the mean-geometry regularizers (hinge separation and
inverse-square repulsion), and the supervised contrastive (SupCon) loss.
"""
import torch
import torch.nn.functional as F

from .config import EMB_DIM, N_CLASSES, DEVICE

MIN_PER_CLASS = 8       # skip a class in a batch if it has too few samples for the test
ANCHOR_SCALE = 6.0      # spacing of the per-class Gaussian centres
MEANS_MARGIN = 6.0      # hinge-separation margin between learnable means
REPULSE_EPS = 1e-6


# --------------------------------------------------------------------------- #
# SIGReg                                                                       #
# --------------------------------------------------------------------------- #
def standard_normal_quantiles(n, device):
    """Quantiles of N(0,1) at plotting positions (k-0.5)/n, k=1..n."""
    p = (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n
    return torch.sqrt(torch.tensor(2.0, device=device)) * torch.erfinv(2.0 * p - 1.0)


def sigreg_loss(z, n_slices=64):
    """
    Sketched Isotropic Gaussian Regularization.

    Project the batch `z` (B x D) onto `n_slices` random unit directions and, for
    each slice, measure the squared 1-D Wasserstein-2 distance between the sorted
    projections and the standard-normal quantiles.  Averaging over random slices
    approximates the deviation of the joint distribution from N(0, I).
    """
    b, d = z.shape
    directions = F.normalize(torch.randn(d, n_slices, device=z.device), dim=0)
    proj = z @ directions
    proj_sorted, _ = torch.sort(proj, dim=0)
    q = standard_normal_quantiles(b, z.device).unsqueeze(1)
    return ((proj_sorted - q) ** 2).mean()


def classwise_sigreg_loss(z, y, means, n_slices=64):
    """SIGReg applied independently to each supervised category in the batch."""
    losses = []
    for c in torch.unique(y):
        mask = y == c
        if mask.sum() < MIN_PER_CLASS:
            continue
        zc = z[mask] - means[c]
        losses.append(sigreg_loss(zc, n_slices=n_slices))
    if not losses:
        return z.new_zeros(())
    return torch.stack(losses).mean()


# --------------------------------------------------------------------------- #
# Class-mean geometry                                                          #
# --------------------------------------------------------------------------- #
def make_anchors(scale=ANCHOR_SCALE):
    """One fixed, well-separated anchor per class (scaled standard basis vectors)."""
    anchors = torch.zeros(N_CLASSES, EMB_DIM, device=DEVICE)
    for c in range(N_CLASSES):
        anchors[c, c] = scale
    return anchors


def _pairwise(means):
    d = torch.cdist(means, means)
    iu = torch.triu_indices(means.size(0), means.size(0), offset=1)
    return d[iu[0], iu[1]]


def separation_loss(means, margin=MEANS_MARGIN):
    """Hinge penalty on pairs of class means closer than `margin` (prevents collapse)."""
    pair_d = _pairwise(means)
    return torch.clamp(margin - pair_d, min=0.0).pow(2).mean()


def repulsion_loss(means):
    """Inverse-square repulsion between every pair of class means (Coulomb-like)."""
    return (1.0 / (_pairwise(means).pow(2) + REPULSE_EPS)).sum()


def shrink_loss(means):
    """Mild pull toward the origin so repulsed means stay finite."""
    return means.pow(2).sum(dim=1).mean()


def mean_geometry(means):
    """Diagnostics: (min, mean) pairwise distance between class means."""
    pair_d = _pairwise(means)
    return pair_d.min().item(), pair_d.mean().item()


# --------------------------------------------------------------------------- #
# Supervised contrastive (supervised SimCLR)                                  #
# --------------------------------------------------------------------------- #
def supcon_loss(feats, labels, temp=0.1):
    """feats: (N, D) L2-normalised; labels: (N,).  Standard SupCon (Khosla 2020)."""
    n = feats.size(0)
    sim = feats @ feats.t() / temp
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(n, dtype=torch.bool, device=feats.device)
    exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
    pos_count = pos.sum(dim=1).clamp(min=1)
    mean_log_prob_pos = (pos * log_prob).sum(dim=1) / pos_count
    return -mean_log_prob_pos.mean()
