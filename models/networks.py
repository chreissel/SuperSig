"""
Networks shared across all experiments.

* MNIST: a convolutional backbone (``ConvBackbone``) and an end-to-end CNN.
* JetClass: permutation-invariant particle-cloud encoders (``JetDeepSets`` and
  a compact self-attention ``JetTransformer``) plus an end-to-end supervised net.

Every backbone maps an input to an ``emb_dim`` embedding, so any of them can be
dropped into the (dataset-agnostic) Lightning modules via the YAML config.
"""
import torch
import torch.nn as nn

from .config import EMB_DIM, N_CLASSES


class ConvBackbone(nn.Module):
    """Shared convolutional feature extractor -> `emb_dim` embedding."""

    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 28 -> 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 14 -> 7
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, emb_dim),
        )

    def forward(self, x):
        return self.head(self.features(x))


class SupervisedCNN(nn.Module):
    """Backbone + classification head, trained end-to-end (baseline)."""

    def __init__(self, emb_dim=EMB_DIM, n_classes=N_CLASSES):
        super().__init__()
        self.backbone = ConvBackbone(emb_dim)
        self.classifier = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        return self.classifier(self.backbone(x))


# --------------------------------------------------------------------------- #
# JetClass encoders (particle clouds -> embedding)                             #
#                                                                             #
# A jet is a set of particles `[B, P, F]` with zero-padded slots.  The mask is  #
# derived from the padding sentinel (all-zero feature rows) so the encoders     #
# stay drop-in for the same Lightning modules used on MNIST.                    #
# --------------------------------------------------------------------------- #
def _particle_mask(x):
    """`[B, P, 1]` float mask: 1 for real particles, 0 for padded slots."""
    return (x.abs().sum(-1, keepdim=True) > 0).float()


class JetDeepSets(nn.Module):
    """
    Deep Sets encoder: a per-particle MLP (phi), masked mean-pooling over
    particles, then a jet-level MLP (rho) -> `emb_dim` embedding.
    """

    def __init__(self, input_dim, emb_dim=EMB_DIM, phi_hidden=(128, 128), rho_hidden=(128,)):
        super().__init__()
        dims = [input_dim, *phi_hidden]
        phi = []
        for a, b in zip(dims[:-1], dims[1:]):
            phi += [nn.Linear(a, b), nn.ReLU()]
        self.phi = nn.Sequential(*phi)

        dims = [phi_hidden[-1], *rho_hidden]
        rho = []
        for a, b in zip(dims[:-1], dims[1:]):
            rho += [nn.Linear(a, b), nn.ReLU()]
        rho += [nn.Linear(dims[-1], emb_dim)]
        self.rho = nn.Sequential(*rho)

    def forward(self, x):                                    # x: [B, P, F]
        mask = _particle_mask(x)
        h = self.phi(x) * mask
        pooled = h.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.rho(pooled)


class JetTransformer(nn.Module):
    """
    Compact particle-cloud transformer (a lightweight, self-contained stand-in
    for the reference's ParticleTransformer): embed particles, prepend a CLS
    token, run masked self-attention, read out the CLS token -> `emb_dim`.
    """

    def __init__(self, input_dim, emb_dim=EMB_DIM, d_model=128, nhead=8,
                 num_layers=4, dim_feedforward=256, dropout=0.0):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(d_model, emb_dim)

    def forward(self, x):                                    # x: [B, P, F]
        real = x.abs().sum(-1) > 0                           # [B, P] True=real
        h = self.embed(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        cls_real = torch.ones(x.size(0), 1, dtype=torch.bool, device=x.device)
        key_padding = ~torch.cat([cls_real, real], dim=1)   # True = ignore
        out = self.encoder(h, src_key_padding_mask=key_padding)
        return self.head(out[:, 0])


class SupervisedJetNet(nn.Module):
    """Jet backbone + linear classifier, trained end-to-end (JetClass baseline)."""

    def __init__(self, input_dim, emb_dim=EMB_DIM, n_classes=N_CLASSES, encoder="deepsets"):
        super().__init__()
        if encoder == "transformer":
            self.backbone = JetTransformer(input_dim, emb_dim)
        else:
            self.backbone = JetDeepSets(input_dim, emb_dim)
        self.classifier = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        return self.classifier(self.backbone(x))
