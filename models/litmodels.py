"""
Lightning modules for every training regime in the study.

Following the phlab-neurips25 convention, all the PyTorch-Lightning boilerplate
lives here: each module receives its network(s) (encoder / classifier) from the
YAML config and only implements the train/val steps and the optimizer.  The
actual objective is delegated to the *unchanged* loss functions in
``models.losses`` -- these modules never re-implement a loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl

from . import losses as _losses
from .config import EMB_DIM, N_CLASSES
from .losses import (
    sigreg_loss,
    classwise_sigreg_loss,
    separation_loss,
    repulsion_loss,
    shrink_loss,
    mean_geometry,
    supcon_loss,
)


def _make_anchors(n_classes, emb_dim):
    """
    Build ``n_classes`` orthogonal anchors of dimension ``emb_dim`` by reusing
    the unchanged ``losses.make_anchors`` verbatim.

    ``make_anchors`` reads the class/embedding size from module-level names; we
    temporarily point those at the requested values (restoring them afterwards)
    so a five-class JetClass run and a ten-class MNIST run both get correctly
    sized anchors without touching the loss implementation.
    """
    prev = (_losses.N_CLASSES, _losses.EMB_DIM)
    _losses.N_CLASSES, _losses.EMB_DIM = n_classes, emb_dim
    try:
        return _losses.make_anchors()
    finally:
        _losses.N_CLASSES, _losses.EMB_DIM = prev


# --------------------------------------------------------------------------- #
# Supervised baseline                                                          #
# --------------------------------------------------------------------------- #
class SupervisedModule(pl.LightningModule):
    """CNN trained end-to-end with categorical cross-entropy (reference)."""

    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# --------------------------------------------------------------------------- #
# Self-supervised SIGReg (invariance + global isotropic-Gaussian)             #
# --------------------------------------------------------------------------- #
class SIGRegSSLModule(pl.LightningModule):
    """
    Two augmented views -> invariance (MSE) + a global isotropic-Gaussian SIGReg
    term on each view.  The loader is expected to yield ``(view1, view2)``.
    """

    def __init__(self, encoder, projector=None, lam=1.0, lr=1e-3):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.lam = lam
        self.lr = lr
        self.save_hyperparameters(ignore=["encoder", "projector"])

    def forward(self, x):
        return self.encoder(x)                              # embedding for downstream eval

    def _project(self, x):
        h = self.encoder(x)
        return self.projector(h) if self.projector is not None else h

    def _step(self, batch, stage):
        v1, v2 = batch
        z1, z2 = self._project(v1), self._project(v2)
        inv = F.mse_loss(z1, z2)
        reg = 0.5 * (sigreg_loss(z1) + sigreg_loss(z2))
        loss = inv + self.lam * reg
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}/inv", inv, on_step=False, on_epoch=True)
        self.log(f"{stage}/sigreg", reg, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# --------------------------------------------------------------------------- #
# Class-conditional SIGReg (fixed anchors / learnable means / repulsion)      #
# --------------------------------------------------------------------------- #
class ClasswiseSIGRegModule(pl.LightningModule):
    """
    Class-conditional SIGReg with a selectable mean-geometry strategy.  The
    loader yields ``(image, label)``.

    ``mode="fixed"``      : anchors frozen (registered as a buffer), no aux term.
    ``mode="learnmeans"`` : means are trainable, hinge separation term.
    ``mode="repulse"``    : means are trainable, inverse-square repulsion + shrink.
    """

    def __init__(self, encoder, mode="fixed", n_classes=N_CLASSES, emb_dim=EMB_DIM,
                 rep_weight=20.0, shrink_weight=0.02, beta_sep=0.5, lr=1e-3):
        super().__init__()
        if mode not in ("fixed", "learnmeans", "repulse"):
            raise ValueError(f"unknown mode {mode!r}")
        self.encoder = encoder
        self.mode = mode
        self.n_classes = n_classes
        self.emb_dim = emb_dim
        self.rep_weight = rep_weight
        self.shrink_weight = shrink_weight
        self.beta_sep = beta_sep
        self.lr = lr

        anchors = _make_anchors(n_classes, emb_dim)
        if mode == "fixed":
            self.register_buffer("means", anchors.clone())
        else:
            self.means = nn.Parameter(anchors.clone())
        self.save_hyperparameters(ignore=["encoder"])

    def forward(self, x):
        return self.encoder(x)

    def _aux(self):
        if self.mode == "learnmeans":
            return self.beta_sep * separation_loss(self.means)
        if self.mode == "repulse":
            return (self.rep_weight * repulsion_loss(self.means)
                    + self.shrink_weight * shrink_loss(self.means))
        return torch.zeros((), device=self.device)

    def _step(self, batch, stage):
        x, y = batch
        z = self.encoder(x)
        reg = classwise_sigreg_loss(z, y, self.means)
        aux = self._aux()
        loss = reg + aux
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}/sigreg", reg, on_step=False, on_epoch=True)
        self.log(f"{stage}/aux", aux, on_step=False, on_epoch=True)
        if stage == "val":
            dmin, dmean = mean_geometry(self.means.detach())
            self.log("val/min_dist", dmin, on_epoch=True)
            self.log("val/mean_dist", dmean, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# --------------------------------------------------------------------------- #
# Supervised contrastive (supervised SimCLR)                                  #
# --------------------------------------------------------------------------- #
class SupConModule(pl.LightningModule):
    """
    Supervised contrastive (SupCon) training.  The loader yields
    ``(view1, view2, label)``; both views are embedded, L2-normalised and passed
    to the SupCon loss with the labels duplicated across the two views.
    """

    def __init__(self, encoder, projector=None, temperature=0.1, lr=1e-3):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.temperature = temperature
        self.lr = lr
        self.save_hyperparameters(ignore=["encoder", "projector"])

    def forward(self, x):
        return self.encoder(x)                              # embedding for downstream eval

    def _step(self, batch, stage):
        v1, v2, y = batch
        h = self.encoder(torch.cat([v1, v2]))
        if self.projector is not None:
            h = self.projector(h)
        z = F.normalize(h, dim=1)
        loss = supcon_loss(z, torch.cat([y, y]), temp=self.temperature)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
