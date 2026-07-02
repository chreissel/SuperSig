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


def default_epochs(quick, full):
    return (1 if quick else full)


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
