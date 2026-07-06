"""
Export frozen-encoder embeddings for the FULL JetClass test split to numpy.

Loads a checkpoint (e.g. the 4-class hold-out-4 supervised-SimCLR model) and runs
its frozen encoder over the *entire* test set of **all five** classes -- including
the held-out Hbb class -- writing one row per jet::

    label, z0, z1, z2, z3

The model and data are reconstructed from the same training config the checkpoint
was produced with (so the encoder architecture matches exactly); the only change
for the export is that the test class list is expanded to all five classes, so the
held-out class is embedded too.

Typical use with the hold-out-4 checkpoint (encoder embedding dim = 4)::

    python experiments/07_export_embeddings.py \
        --config configs/jetclass_supsimclr_holdout4.yaml \
        --ckpt   runs/jetclass_supsimclr_holdout4/checkpoints/last.ckpt \
        --out    holdout4_test_embeddings.npz

Output (.npz) contains:
    labels      : int64  [N]        class index (0..4; 4 = Hbb, the held-out class)
    embeddings  : float32[N, D]      the D-dim embedding (D = 4 for the hold-out-4 model)
    table       : float32[N, 1 + D]  columns [label, z0, z1, ..., z(D-1)]
    columns     : str   [1 + D]      column names for `table`
If --out ends in .npy, only `table` is written.
"""
import os
import sys
import argparse

# This script never trains or logs.
os.environ.setdefault("WANDB_MODE", "disabled")

# Make the repo root importable (also lets jsonargparse resolve the config's
# class_path entries).
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import torch
from lightning.pytorch.cli import LightningCLI

from common import frozen_encoder
from models.config import DEVICE
from data.jetclass_data import DEFAULT_CLASSES


def build_from_config(config):
    """Instantiate the model + datamodule from a training config (no trainer run)."""
    # Hide this script's own CLI flags from LightningCLI (it parses `args=` only).
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        cli = LightningCLI(
            args=["--config", config, "--trainer.logger=false"],
            run=False, save_config_callback=None,
        )
    finally:
        sys.argv = argv
    return cli.model, cli.datamodule


@torch.no_grad()
def embed_split(encoder, loader, max_jets=None, log_every=50000):
    """Run ``encoder`` over every jet in ``loader``; return (embeddings, labels)."""
    encoder.eval()
    embs, labels, n = [], [], 0
    for x, y in loader:
        embs.append(encoder(x.to(DEVICE)).cpu().numpy().astype("float32"))
        labels.append(np.asarray(y).astype("int64"))
        n += len(labels[-1])
        if n // log_every != (n - len(labels[-1])) // log_every:
            print(f"  embedded {n} jets...", flush=True)
        if max_jets is not None and n >= max_jets:
            break
    embs = np.concatenate(embs)
    labels = np.concatenate(labels)
    if max_jets is not None:
        embs, labels = embs[:max_jets], labels[:max_jets]
    return embs, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="training config the checkpoint was produced with")
    ap.add_argument("--ckpt", required=True, help="checkpoint (.ckpt) to load")
    ap.add_argument("--out", default=None,
                    help="output path (.npz default; .npy writes only the [label, z...] table)")
    ap.add_argument("--data-dir", default=None,
                    help="JetClass ROOT directory (defaults to the config / $JETCLASS_DIR)")
    ap.add_argument("--max-files-per-class", type=int, default=None,
                    help="cap test ROOT files per class (default: all -> full test set)")
    ap.add_argument("--max-jets", type=int, default=0,
                    help="cap total jets embedded (default 0 = all)")
    ap.add_argument("--batch-size", type=int, default=None, help="override the config batch size")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers (default 0; >0 needs more RAM)")
    args = ap.parse_args()

    model, dm = build_from_config(args.config)

    # Restore the trained weights (weights_only=False: Lightning ckpts pickle more
    # than tensors, and torch>=2.6 defaults to rejecting that).
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    print(f"  loaded checkpoint {args.ckpt}")

    # Expand the test set to ALL five classes so the held-out class is embedded too;
    # the label space is the data config's 5-class order regardless of what the
    # encoder was trained on, so Hbb comes out as label 4.
    dm.class_names = list(DEFAULT_CLASSES)
    if args.data_dir is not None:
        dm.data_dir = args.data_dir
    dm.max_files_per_class = args.max_files_per_class
    dm.num_workers = args.num_workers
    dm.loader_kwargs["num_workers"] = args.num_workers
    if args.batch_size is not None:
        dm.batch_size = args.batch_size
        dm.loader_kwargs["batch_size"] = args.batch_size
    dm.setup()
    print(f"  test classes: {dm.class_names}")

    encoder = frozen_encoder(model)
    max_jets = None if args.max_jets in (0, None) else args.max_jets
    embs, labels = embed_split(encoder, dm.test_dataloader(), max_jets=max_jets)

    D = embs.shape[1]
    uniq, counts = np.unique(labels, return_counts=True)
    per_class = {DEFAULT_CLASSES[int(u)]: int(c) for u, c in zip(uniq, counts)}
    print(f"  embedded {len(embs)} jets, embedding dim = {D}")
    print(f"  per-class counts: {per_class}")

    table = np.column_stack([labels.astype("float32"), embs]).astype("float32")
    columns = np.array(["label"] + [f"z{i}" for i in range(D)])

    cfg_tag = os.path.splitext(os.path.basename(args.config))[0]
    out = args.out or f"{cfg_tag}_test_embeddings.npz"
    if out.endswith(".npy"):
        np.save(out, table)
    else:
        np.savez(out, labels=labels, embeddings=embs, table=table, columns=columns)
    print(f"  saved {out}  (columns: {', '.join(columns)})")


if __name__ == "__main__":
    main()
