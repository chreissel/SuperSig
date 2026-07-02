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

import torch
import lightning as pl

from models.config import DEVICE
from models.networks import (ConvBackbone, SupervisedCNN, ParticleTransformerModel,
                             SupervisedJetNet, MLP)
from models.config import EMB_DIM
from data.datasets import (
    MNISTDataModule, ClasswiseMNISTDataModule, TwoViewMNISTDataModule,
    JetClassDataModule, JetClassClasswiseDataModule, JetClassTwoViewDataModule,
)
from data.jetclass_data import N_FEATURES as JET_FEATURES, DEFAULT_CLASSES as JET_CLASSES
from models.config import N_CLASSES as MNIST_N_CLASSES

DATASETS = ("mnist", "jetclass")


def n_classes(dataset):
    """Number of classes for the dataset (JetClass uses the 5-class subset)."""
    return MNIST_N_CLASSES if dataset == "mnist" else len(JET_CLASSES)


def default_epochs(quick, full):
    return (1 if quick else full)


# --------------------------------------------------------------------------- #
# Dataset-aware factories: the same test suite, either dataset                 #
# --------------------------------------------------------------------------- #
def make_encoder(dataset):
    """The backbone/encoder appropriate for the dataset (both -> EMB_DIM)."""
    return ConvBackbone() if dataset == "mnist" else ParticleTransformerModel(input_dim=JET_FEATURES)


def make_projector(dataset):
    """Projection head for the contrastive modules (JetClass only; mirrors reference)."""
    return None if dataset == "mnist" else MLP(EMB_DIM, [EMB_DIM], EMB_DIM)


def make_supervised_net(dataset):
    """End-to-end supervised network (backbone + classifier) for the dataset."""
    if dataset == "mnist":
        return SupervisedCNN()
    return SupervisedJetNet(input_dim=JET_FEATURES, n_classes=n_classes(dataset), encoder="part")


def plain_dm(dataset, quick, batch_size, data_dir=None):
    """Plain ``(x, y)`` DataModule (supervised training + probes)."""
    if dataset == "mnist":
        return MNISTDataModule(quick=quick, batch_size=batch_size)
    return JetClassDataModule(quick=quick, batch_size=batch_size, data_dir=data_dir)


def classwise_dm(dataset, quick, batch_size, holdout=None, data_dir=None):
    """``(x, y)`` DataModule that can drop a held-out class from training."""
    if dataset == "mnist":
        return ClasswiseMNISTDataModule(quick=quick, holdout=holdout, batch_size=batch_size)
    return JetClassClasswiseDataModule(quick=quick, holdout=holdout, batch_size=batch_size,
                                       data_dir=data_dir)


def twoview_dm(dataset, quick, batch_size, labeled, holdout=None, data_dir=None):
    """Two-view DataModule (SIGReg-SSL / SupCon), optionally labelled / held-out."""
    if dataset == "mnist":
        return TwoViewMNISTDataModule(quick=quick, labeled=labeled, holdout=holdout,
                                      batch_size=batch_size)
    return JetClassTwoViewDataModule(quick=quick, labeled=labeled, holdout=holdout,
                                     batch_size=batch_size, data_dir=data_dir)


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
        state = torch.load(ckpt, map_location="cpu")
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
