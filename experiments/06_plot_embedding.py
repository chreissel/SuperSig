"""
Plotting-only: visualise a trained embedding on the JetClass *test* split.

No training and no linear probe -- this loads a checkpoint, embeds jets from a
single test file per class, and draws a corner plot of the (frozen) encoder
embedding coloured by class.

Everything about the model and the data (encoder architecture, projector,
class list, data config) is read straight from the *same* training config, so
the reconstructed network matches the checkpoint exactly.  Typical use with the
supervised-SimCLR configs::

    python experiments/06_plot_embedding.py \
        --config configs/jetclass_supsimclr.yaml \
        --ckpt   runs/jetclass_supsimclr/checkpoints/last.ckpt

    python experiments/06_plot_embedding.py \
        --config configs/jetclass_supsimclr_holdout4.yaml \
        --ckpt   runs/jetclass_supsimclr_holdout4/checkpoints/last.ckpt

By default only one ROOT file per class is read (``--max-files-per-class 1``)
and at most ``--max-jets`` jets are embedded, so the plot is quick to produce.
"""
import os
import sys
import argparse

# Quiet W&B / offline: this script never trains or logs.
os.environ.setdefault("WANDB_MODE", "disabled")

# Make the repo root importable (needed both for our imports and for
# jsonargparse to resolve the ``class_path`` entries in the config).
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import torch
from lightning.pytorch.cli import LightningCLI

from common import frozen_encoder
from models.config import plot_path, DEVICE
from data.jetclass_data import DEFAULT_CLASSES
from utils.plotting import save_corner


def build_from_config(config):
    """Instantiate the model + datamodule from a training config (no trainer run)."""
    cli = LightningCLI(
        args=["--config", config, "--trainer.logger=false"],
        run=False, save_config_callback=None,
    )
    return cli.model, cli.datamodule


@torch.no_grad()
def collect_embeddings(encoder, loader, max_jets=None):
    """Embed jets from ``loader`` (early-stopping once ``max_jets`` are seen)."""
    encoder.eval()
    embs, labels, n = [], [], 0
    for x, y in loader:
        embs.append(encoder(x.to(DEVICE)).cpu().numpy())
        labels.append(np.asarray(y))
        n += len(y)
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
    ap.add_argument("--data-dir", default=None,
                    help="JetClass ROOT directory (defaults to the config / $JETCLASS_DIR)")
    ap.add_argument("--max-files-per-class", type=int, default=1,
                    help="test ROOT files per class to read (default 1)")
    ap.add_argument("--max-jets", type=int, default=5000,
                    help="cap the number of jets embedded for the plot (default 5000; 0 = all)")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers (default 0; >0 needs more RAM)")
    ap.add_argument("--out", default=None, help="output PNG path (default: plots/<config>_embedding.png)")
    args = ap.parse_args()

    model, dm = build_from_config(args.config)

    # Restore the trained weights (weights_only=False: Lightning ckpts pickle more
    # than tensors, and torch>=2.6 defaults to rejecting that).
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    print(f"  loaded checkpoint {args.ckpt}")

    # Point the datamodule at one test file per class (fast, single stream).
    if args.data_dir is not None:
        dm.data_dir = args.data_dir
    dm.max_files_per_class = args.max_files_per_class
    dm.num_workers = args.num_workers
    dm.loader_kwargs["num_workers"] = args.num_workers
    dm.setup()

    encoder = frozen_encoder(model)
    max_jets = None if args.max_jets in (0, None) else args.max_jets
    embs, labels = collect_embeddings(encoder, dm.test_dataloader(), max_jets=max_jets)

    names = [DEFAULT_CLASSES[i] for i in sorted(np.unique(labels))]
    print(f"  embedded {len(embs)} jets, dim={embs.shape[1]}, classes present: {names}")

    cfg_tag = os.path.splitext(os.path.basename(args.config))[0]
    out = args.out or plot_path(f"{cfg_tag}_embedding.png")
    # Reference-style corner plot (phlab-neurips25 make_corner), the same style
    # it logs to W&B: DEFAULT_CLASSES gives the per-index class names for the legend.
    save_corner(embs, labels, out, label_names=DEFAULT_CLASSES,
                title=f"{cfg_tag} test embedding")


if __name__ == "__main__":
    main()
