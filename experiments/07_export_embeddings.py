"""
Export frozen-encoder embeddings for the JetClass test split to numpy, one npz
per ROOT file.

Loads a checkpoint (e.g. the 4-class hold-out-4 supervised-SimCLR model) and runs
its frozen encoder over the test set of **all five** classes -- including the
held-out Hbb class -- but writes the result of **each ROOT file to its own npz**.
Processing one file at a time keeps memory bounded and avoids a single enormous
write, so it scales to the full ``test_20M`` split without I/O trouble.

Each per-file npz holds one row per jet::

    label, z0, z1, z2, z3

The model and data are reconstructed from the same training config the checkpoint
was produced with (so the encoder architecture matches exactly); the only change
for the export is that the test class list is expanded to all five classes, so the
held-out class is embedded too.

Typical use with the hold-out-4 checkpoint (encoder embedding dim = 4)::

    python experiments/07_export_embeddings.py \
        --config  configs/jetclass_supsimclr_holdout4.yaml \
        --ckpt    runs/jetclass_supsimclr_holdout4/checkpoints/last.ckpt \
        --out-dir holdout4_test_embeddings/

Each output file ``<out-dir>/<root-file-stem>.npz`` contains:
    labels      : int64  [n]        class index (0..4; 4 = Hbb, the held-out class)
    embeddings  : float32[n, D]      the D-dim embedding (D = 4 for the hold-out-4 model)
    table       : float32[n, 1 + D]  columns [label, z0, z1, ..., z(D-1)]
    columns     : str   [1 + D]      column names for `table`
"""
import os
import sys
import glob
import argparse

# This script never trains or logs.
os.environ.setdefault("WANDB_MODE", "disabled")

# Make the repo root importable (also lets jsonargparse resolve the config's
# class_path entries).
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import torch
from torch.utils.data import DataLoader
from lightning.pytorch.cli import LightningCLI

from common import frozen_encoder
from models.config import DEVICE
from data.jetclass_data import DEFAULT_CLASSES
import data.jetclass_data as jc


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


def file_loader(dm, cls, path):
    """A DataLoader that streams the single ROOT ``path`` (class ``cls``)."""
    ds = jc.make_iter_dataset({cls: [path]}, dm.data_config, for_training=False,
                              fetch_step=dm.fetch_step, name=f"test:{cls}")
    kwargs = dict(dm.loader_kwargs)
    if kwargs.get("num_workers", 0) > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(jc.JetClassAdapter(ds, mode="plain"), **kwargs)


@torch.no_grad()
def embed_loader(encoder, loader, max_jets=None):
    """Run ``encoder`` over every jet in ``loader``; return (embeddings, labels)."""
    encoder.eval()
    embs, labels, n = [], [], 0
    for x, y in loader:
        embs.append(encoder(x.to(DEVICE)).cpu().numpy().astype("float32"))
        labels.append(np.asarray(y).astype("int64"))
        n += len(labels[-1])
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
    ap.add_argument("--out-dir", default=None,
                    help="directory for the per-file npz outputs (default: <config>_test_embeddings/)")
    ap.add_argument("--data-dir", default=None,
                    help="JetClass ROOT directory (defaults to the config / $JETCLASS_DIR)")
    ap.add_argument("--max-files-per-class", type=int, default=None,
                    help="cap test ROOT files per class (default: all -> full test set)")
    ap.add_argument("--max-jets-per-file", type=int, default=0,
                    help="cap jets embedded per file (default 0 = all)")
    ap.add_argument("--batch-size", type=int, default=None, help="override the config batch size")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers (default 0; >0 needs more RAM)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-embed and overwrite npz files that already exist (default: skip)")
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
    if args.data_dir is not None:
        dm.data_dir = args.data_dir
    dm.num_workers = args.num_workers
    dm.loader_kwargs["num_workers"] = args.num_workers
    if args.batch_size is not None:
        dm.batch_size = args.batch_size
        dm.loader_kwargs["batch_size"] = args.batch_size

    file_dict = jc.build_file_dict(dm.data_dir, "test", list(DEFAULT_CLASSES),
                                   max_files_per_class=args.max_files_per_class)
    if not file_dict:
        raise FileNotFoundError(
            f"No JetClass test ROOT files under {dm.data_dir!r} "
            f"(expected a test_20M/ subfolder of *.root files).")

    cfg_tag = os.path.splitext(os.path.basename(args.config))[0]
    out_dir = args.out_dir or f"{cfg_tag}_test_embeddings"
    os.makedirs(out_dir, exist_ok=True)

    encoder = frozen_encoder(model)
    max_jets = None if args.max_jets_per_file in (0, None) else args.max_jets_per_file
    n_files = sum(len(v) for v in file_dict.values())
    print(f"  classes: {list(file_dict)}  |  {n_files} test file(s) -> {out_dir}/")

    total_jets, done = 0, 0
    for cls in DEFAULT_CLASSES:
        for path in file_dict.get(cls, []):
            done += 1
            stem = os.path.splitext(os.path.basename(path))[0]
            out = os.path.join(out_dir, f"{stem}.npz")
            if os.path.exists(out) and not args.overwrite:
                print(f"  [{done}/{n_files}] skip {stem} (exists)")
                continue

            embs, labels = embed_loader(encoder, file_loader(dm, cls, path), max_jets=max_jets)
            D = embs.shape[1]
            table = np.column_stack([labels.astype("float32"), embs]).astype("float32")
            columns = np.array(["label"] + [f"z{i}" for i in range(D)])
            np.savez(out, labels=labels, embeddings=embs, table=table, columns=columns)
            total_jets += len(embs)
            print(f"  [{done}/{n_files}] {cls:5s} {stem}: {len(embs)} jets -> {out}")

    print(f"  done: embedded {total_jets} jets across {n_files} file(s) into {out_dir}/  "
          f"(each npz: labels, embeddings, table[label, z0, z1, ...], columns)")


if __name__ == "__main__":
    main()
