"""
Shared helpers for the downstream-evaluation scripts.

Every script trains an embedding with a Lightning module (a short in-script
``Trainer.fit`` so the examples stay runnable end-to-end), or loads one from a
checkpoint, then hands the frozen encoder to the plain-PyTorch probe/plot
helpers in :mod:`utils.eval` and :mod:`utils.plotting`.

The canonical way to train these embeddings is still the CLI, e.g.::

    python cli.py fit --config configs/mnist_supcon.yaml

and the resulting ``last.ckpt`` can be fed back here with ``--ckpt``.
"""
import os
import sys

# Make the repo root importable when a script is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re

import torch
import lightning as pl

from models.config import DEVICE
from models.networks import (ConvBackbone, SupervisedCNN, ParticleTransformerModel,
                             SupervisedJetNet, MLP)
from data.datasets import (
    MNISTDataModule, ClasswiseMNISTDataModule, TwoViewMNISTDataModule,
    JetClassDataModule, JetClassClasswiseDataModule, JetClassTwoViewDataModule,
)
from data.jetclass_data import N_FEATURES as JET_FEATURES, DEFAULT_CLASSES as JET_CLASSES
from models.config import N_CLASSES as MNIST_N_CLASSES

DATASETS = ("mnist", "jetclass")

# Default embedding dim for *in-script* training (when no checkpoint is loaded).
# These mirror the YAML configs; a --ckpt overrides them via emb_dim_from_ckpt.
DEFAULT_EMB_DIM = {"mnist": 16, "jetclass": 8}


def n_classes(dataset):
    """Number of classes for the dataset (JetClass uses the 5-class subset)."""
    return MNIST_N_CLASSES if dataset == "mnist" else len(JET_CLASSES)


def default_epochs(quick, full):
    return (1 if quick else full)


# --------------------------------------------------------------------------- #
# Dataset-aware factories: the same test suite, either dataset                 #
# --------------------------------------------------------------------------- #
def make_encoder(dataset, emb_dim=None):
    """The backbone/encoder appropriate for the dataset, producing an ``emb_dim``
    embedding.  ``emb_dim`` defaults to the dataset's in-script default; pass the
    value read from a checkpoint (see ``emb_dim_from_ckpt``) to match a loaded model."""
    emb_dim = emb_dim or DEFAULT_EMB_DIM[dataset]
    if dataset == "mnist":
        return ConvBackbone(emb_dim=emb_dim)
    return ParticleTransformerModel(input_dim=JET_FEATURES, emb_dim=emb_dim)


def make_projector(dataset, emb_dim):
    """Projection head for the contrastive modules (JetClass only; mirrors reference)."""
    return None if dataset == "mnist" else MLP(emb_dim, [emb_dim], emb_dim)


def emb_dim_from_ckpt(ckpt_path):
    """Read the encoder's embedding dimension straight from a checkpoint.

    Returns the out-features of the encoder's final linear -- ParticleTransformer
    (``encoder.model.fc.<n>.weight``) or ConvBackbone (``encoder.head.<n>.weight``)
    -- or ``None`` if it can't be determined (caller falls back to the default)."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    for pat in (r"^encoder\.model\.fc\.(\d+)\.weight$",   # ParticleTransformer
                r"^encoder\.head\.(\d+)\.weight$"):        # ConvBackbone
        hits = [(int(m.group(1)), k) for k in sd for m in [re.match(pat, k)] if m]
        if hits:
            return sd[max(hits)[1]].shape[0]
    return None


def make_supervised_net(dataset):
    """End-to-end supervised network (backbone + classifier) for the dataset."""
    if dataset == "mnist":
        return SupervisedCNN()
    return SupervisedJetNet(input_dim=JET_FEATURES, n_classes=n_classes(dataset))


def _jet_kwargs(quick, batch_size, data_dir, num_workers, max_files_per_class):
    # JetClass streams via weaver; default to a single in-process worker
    # (num_workers=0) so several parallel weaver streams can't exhaust memory on a
    # constrained node.  `max_files_per_class` bounds how much of the split the
    # downstream probe/eval reads (None = everything).
    kw = dict(quick=quick, batch_size=batch_size, data_dir=data_dir,
              num_workers=(0 if num_workers is None else num_workers))
    if max_files_per_class is not None:
        kw["max_files_per_class"] = max_files_per_class
    return kw


def plain_dm(dataset, quick, batch_size, data_dir=None, num_workers=None, max_files_per_class=None):
    """Plain ``(x, y)`` DataModule (supervised training + probes)."""
    if dataset == "mnist":
        return MNISTDataModule(quick=quick, batch_size=batch_size)
    return JetClassDataModule(**_jet_kwargs(quick, batch_size, data_dir, num_workers, max_files_per_class))


def classwise_dm(dataset, quick, batch_size, holdout=None, data_dir=None,
                 num_workers=None, max_files_per_class=None):
    """``(x, y)`` DataModule that can drop a held-out class from training."""
    if dataset == "mnist":
        return ClasswiseMNISTDataModule(quick=quick, holdout=holdout, batch_size=batch_size)
    return JetClassClasswiseDataModule(
        holdout=holdout, **_jet_kwargs(quick, batch_size, data_dir, num_workers, max_files_per_class))


def twoview_dm(dataset, quick, batch_size, labeled, holdout=None, data_dir=None,
               num_workers=None, max_files_per_class=None):
    """Two-view DataModule (SIGReg-SSL / SupCon), optionally labelled / held-out."""
    if dataset == "mnist":
        return TwoViewMNISTDataModule(quick=quick, labeled=labeled, holdout=holdout,
                                      batch_size=batch_size)
    return JetClassTwoViewDataModule(
        labeled=labeled, holdout=holdout,
        **_jet_kwargs(quick, batch_size, data_dir, num_workers, max_files_per_class))


def outfile(dataset, name):
    """Figure filename, suffixed per dataset so runs don't overwrite each other."""
    if dataset == "mnist":
        return name
    root, ext = os.path.splitext(name)
    return f"{root}_{dataset}{ext}"


CLASS_WORD = {"mnist": "digit", "jetclass": "class"}


def make_trainer(max_epochs, quick):
    """A minimal, logger-less trainer for the in-script example trainings."""
    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        limit_val_batches=(2 if quick else 1.0),
        enable_model_summary=False,
    )


def fit_or_load(module, datamodule, max_epochs, quick, ckpt=None):
    """Train ``module`` on ``datamodule`` for ``max_epochs`` epochs, or load ``ckpt``."""
    if ckpt is not None:
        # weights_only=False: Lightning checkpoints also pickle hyperparameters etc.
        # (torch>=2.6 defaults weights_only=True, which rejects them).
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        module.load_state_dict(state["state_dict"] if "state_dict" in state else state)
        print(f"  loaded checkpoint {ckpt}")
    else:
        make_trainer(max_epochs, quick).fit(module, datamodule)
    return module


def frozen_encoder(module):
    """Return the trained encoder, on DEVICE and in eval mode, ready for probing."""
    encoder = module.encoder if hasattr(module, "encoder") else module.model.backbone
    encoder.to(DEVICE).eval()
    return encoder
