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
# JetClass encoders (particle clouds -> embedding)                             #
#                                                                             #
# A jet is a set of particles `[B, P, C]` with zero-padded slots.  Channels    #
# 0:input_dim are the (standardized) per-particle features; channels           #
# input_dim:input_dim+4 are the raw (px,py,pz,E) vectors used by ParT.  The     #
# mask is derived from the padding sentinel so the encoders stay drop-in for    #
# the same Lightning modules used on MNIST.                                     #
# --------------------------------------------------------------------------- #
def _particle_mask(x):
    """`[B, P, 1]` float mask: 1 for real particles, 0 for padded slots."""
    return (x.abs().sum(-1, keepdim=True) > 0).float()


class JetDeepSets(nn.Module):
    """
    Deep Sets encoder: a per-particle MLP (phi), masked mean-pooling over
    particles, then a jet-level MLP (rho) -> `emb_dim` embedding.  Uses only the
    first `input_dim` feature channels (ignores the trailing vector channels).
    """

    def __init__(self, input_dim, emb_dim=EMB_DIM, phi_hidden=(128, 128), rho_hidden=(128,)):
        super().__init__()
        self.input_dim = input_dim
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

    def forward(self, x):                                    # x: [B, P, C]
        x = x[..., :self.input_dim]
        mask = _particle_mask(x)
        h = self.phi(x) * mask
        pooled = h.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.rho(pooled)


class JetTransformer(nn.Module):
    """
    Compact particle-cloud transformer (a lightweight self-attention stand-in):
    embed particles, prepend a CLS token, run masked self-attention, read out the
    CLS token -> `emb_dim`.  Uses the first `input_dim` feature channels.
    """

    def __init__(self, input_dim, emb_dim=EMB_DIM, d_model=128, nhead=8,
                 num_layers=4, dim_feedforward=256, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.embed = nn.Linear(input_dim, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(d_model, emb_dim)

    def forward(self, x):                                    # x: [B, P, C]
        x = x[..., :self.input_dim]
        real = x.abs().sum(-1) > 0                           # [B, P] True=real
        h = self.embed(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        cls_real = torch.ones(x.size(0), 1, dtype=torch.bool, device=x.device)
        key_padding = ~torch.cat([cls_real, real], dim=1)   # True = ignore
        out = self.encoder(h, src_key_padding_mask=key_padding)
        return self.head(out[:, 0])


class ParticleTransformerModel(nn.Module):
    """
    Wrapper around the vendored ParticleTransformer (ParT), mirroring the
    reference.  Splits the packed jet tensor into features / (px,py,pz,E) vectors /
    mask, converts to ParT's channels-first layout, and returns an `emb_dim`
    embedding (ParT's `num_classes` output = the contrastive space).
    """

    def __init__(self, input_dim=17, emb_dim=EMB_DIM, pair_input_dim=4,
                 embed_dims=(128, 512, 128), pair_embed_dims=(64, 64, 64),
                 num_heads=8, num_layers=8, num_cls_layers=2, fc_params=((128, 0.0),),
                 **kwargs):
        super().__init__()
        from .parT import ParticleTransformer
        self.input_dim = input_dim
        self.pair_input_dim = pair_input_dim
        self.model = ParticleTransformer(
            input_dim=input_dim, num_classes=emb_dim, pair_input_dim=pair_input_dim,
            embed_dims=list(embed_dims), pair_embed_dims=list(pair_embed_dims),
            num_heads=num_heads, num_layers=num_layers, num_cls_layers=num_cls_layers,
            fc_params=[list(p) for p in fc_params], **kwargs)

    def forward(self, x):                                    # x: [B, P, C]
        feats = x[..., :self.input_dim].transpose(1, 2)             # [B, F, P]
        vectors = x[..., self.input_dim:self.input_dim + self.pair_input_dim].transpose(1, 2)
        mask = (x.abs().sum(-1) > 0).unsqueeze(1).float()          # [B, 1, P]
        return self.model(feats, v=vectors, mask=mask)


class SupervisedJetNet(nn.Module):
    """Jet backbone + linear classifier, trained end-to-end (JetClass baseline)."""

    def __init__(self, input_dim, emb_dim=EMB_DIM, n_classes=N_CLASSES, encoder="part"):
        super().__init__()
        if encoder == "part":
            self.backbone = ParticleTransformerModel(input_dim=input_dim, emb_dim=emb_dim)
        elif encoder == "transformer":
            self.backbone = JetTransformer(input_dim, emb_dim)
        else:
            self.backbone = JetDeepSets(input_dim, emb_dim)
        self.classifier = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        return self.classifier(self.backbone(x))
