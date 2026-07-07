"""
Networks shared across all experiments.

* MNIST: a convolutional backbone (``ConvBackbone``) and an end-to-end CNN.
* JetClass: the ParticleTransformer encoder (``ParticleTransformerModel``) plus an
  end-to-end supervised net.

Every backbone maps an input to an ``emb_dim`` embedding, so any of them can be
dropped into the (dataset-agnostic) Lightning modules via the YAML config.
"""
import torch.nn as nn


class ConvBackbone(nn.Module):
    """Shared convolutional feature extractor -> `emb_dim` embedding."""

    def __init__(self, emb_dim=16):
        super().__init__()
        self.emb_dim = emb_dim
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

    def __init__(self, emb_dim=16, n_classes=10):
        super().__init__()
        self.backbone = ConvBackbone(emb_dim)
        self.classifier = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        return self.classifier(self.backbone(x))


# --------------------------------------------------------------------------- #
# Projector head                                                               #
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """Simple MLP, e.g. the SimCLR/SupCon projection head (mirrors the reference)."""

    def __init__(self, input_dim, hidden_dims=(), output_dim=None, activation="relu"):
        super().__init__()
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation]
        output_dim = output_dim or input_dim
        layers, cur = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(cur, h), act()]
            cur = h
        layers += [nn.Linear(cur, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# JetClass encoder (particle cloud -> embedding)                               #
#                                                                             #
# A jet is a set of particles `[B, P, C]` with zero-padded slots.  Channels    #
# 0:input_dim are the (standardized) per-particle features; channels           #
# input_dim:input_dim+4 are the raw (px,py,pz,E) vectors used by ParT.          #
# --------------------------------------------------------------------------- #
class ParticleTransformerModel(nn.Module):
    """
    Wrapper around the vendored ParticleTransformer (ParT), mirroring the
    reference.  Splits the packed jet tensor into features / (px,py,pz,E) vectors /
    mask, converts to ParT's channels-first layout, and returns an `emb_dim`
    embedding (ParT's `num_classes` output = the contrastive space).
    """

    def __init__(self, input_dim=17, emb_dim=8, pair_input_dim=4,
                 embed_dims=(128, 512, 128), pair_embed_dims=(64, 64, 64),
                 num_heads=8, num_layers=8, num_cls_layers=2, fc_params=((128, 0.0),),
                 **kwargs):
        super().__init__()
        from .parT import ParticleTransformer
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.pair_input_dim = pair_input_dim
        self.model = ParticleTransformer(
            input_dim=input_dim, num_classes=emb_dim, pair_input_dim=pair_input_dim,
            embed_dims=list(embed_dims), pair_embed_dims=list(pair_embed_dims),
            num_heads=num_heads, num_layers=num_layers, num_cls_layers=num_cls_layers,
            fc_params=[list(p) for p in fc_params], **kwargs)

    def forward(self, x):                                    # x: [B, P, C]
        feats = x[..., :self.input_dim].transpose(1, 2)             # [B, F, P]
        mi = self.input_dim + self.pair_input_dim
        vectors = x[..., self.input_dim:mi].transpose(1, 2)        # [B, 4, P]
        if x.shape[-1] > mi:                                        # explicit mask channel
            mask = x[..., mi].unsqueeze(1).float()                 # [B, 1, P]
        else:                                                      # fall back to nonzero
            mask = (x.abs().sum(-1) > 0).unsqueeze(1).float()
        return self.model(feats, v=vectors, mask=mask)


class SupervisedJetNet(nn.Module):
    """ParticleTransformer backbone + linear classifier, end-to-end (JetClass baseline)."""

    def __init__(self, input_dim, emb_dim=8, n_classes=5):
        super().__init__()
        self.backbone = ParticleTransformerModel(input_dim=input_dim, emb_dim=emb_dim)
        self.classifier = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        return self.classifier(self.backbone(x))
