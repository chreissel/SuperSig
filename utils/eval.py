"""
Downstream-evaluation helpers: frozen-backbone linear probes and collectors.

These are deliberately kept *out* of the Lightning code paths.  As in the
reference repo, embeddings are trained with Lightning (``python cli.py fit
...``) and the downstream analysis (linear probe -> ROC / corner) is plain
PyTorch run from the ``experiments/`` scripts.
"""
import numpy as np
import torch
import torch.nn.functional as F

from models.config import DEVICE, HOLDOUT


# --------------------------------------------------------------------------- #
# Frozen-backbone linear probes                                               #
# --------------------------------------------------------------------------- #
def _freeze(backbone):
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()


def train_linear_probe(backbone, head, loader, epochs, lr=1e-3):
    """Freeze backbone, train a multi-class linear head with cross-entropy."""
    _freeze(backbone)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for ep in range(epochs):
        tot, correct, run = 0, 0, 0.0
        head.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                z = backbone(x)
            opt.zero_grad()
            logits = head(z)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            run += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            tot += x.size(0)
        print(f"  [linear probe] epoch {ep+1}/{epochs}  loss={run/tot:.4f}  acc={correct/tot:.4f}")


def train_binary_probe(backbone, head, loader, epochs, positive=HOLDOUT, lr=1e-3):
    """Freeze backbone, train a 2-way (positive vs rest) linear head."""
    _freeze(backbone)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for ep in range(epochs):
        tot, correct, run = 0, 0, 0.0
        head.train()
        for x, y in loader:
            x = x.to(DEVICE)
            yb = (y == positive).long().to(DEVICE)
            with torch.no_grad():
                z = backbone(x)
            opt.zero_grad()
            logits = head(z)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            run += loss.item() * x.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            tot += x.size(0)
        print(f"  [binary probe] epoch {ep+1}/{epochs}  loss={run/tot:.4f}  acc={correct/tot:.4f}")


# --------------------------------------------------------------------------- #
# Evaluation collectors                                                        #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_probs(forward_fn, loader):
    probs, labels = [], []
    for x, y in loader:
        p = F.softmax(forward_fn(x.to(DEVICE)), dim=1)
        probs.append(p.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


@torch.no_grad()
def collect_binary_scores(backbone, head, loader, positive=HOLDOUT):
    backbone.eval(); head.eval()
    scores, labels = [], []
    for x, y in loader:
        p = F.softmax(head(backbone(x.to(DEVICE))), dim=1)[:, 1]
        scores.append(p.cpu().numpy())
        labels.append((y == positive).long().numpy())
    return np.concatenate(scores), np.concatenate(labels)


@torch.no_grad()
def collect_embeddings(backbone, loader):
    backbone.eval()
    embs, labels = [], []
    for x, y in loader:
        embs.append(backbone(x.to(DEVICE)).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(embs), np.concatenate(labels)
