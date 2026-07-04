"""Self-supervised SIGReg (invariance + global isotropic Gaussian), frozen linear probe.

Works on either dataset via --dataset {mnist,jetclass}.  Trains via the Lightning
`SIGRegSSLModule` (or loads --ckpt), freezes the encoder, fits a linear probe and
draws ROC + corner plots.  Canonical:
    python cli.py fit --config configs/mnist_sigreg_ssl.yaml
    python cli.py fit --config configs/jetclass_sigreg_ssl.yaml
"""
import argparse
import numpy as np
import torch
import torch.nn as nn

from common import (default_epochs, fit_or_load, frozen_encoder,
                    make_encoder, make_projector, plain_dm, twoview_dm, outfile,
                    n_classes, DATASETS)
from models.config import plot_path, EMB_DIM, DEVICE
from models.litmodels import SIGRegSSLModule
from utils.eval import train_linear_probe, collect_probs, collect_embeddings
from utils.plotting import plot_roc, plot_corner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="mnist")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ssl-epochs", type=int, default=None)
    ap.add_argument("--probe-epochs", type=int, default=None)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--data-dir", type=str, default=None,
                    help="JetClass ROOT directory (falls back to $JETCLASS_DIR)")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="JetClass DataLoader workers (default 0; >0 needs more RAM)")
    ap.add_argument("--max-files-per-class", type=int, default=None,
                    help="cap JetClass ROOT files per class for the probe/eval (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ssl_ep = args.ssl_epochs or default_epochs(args.quick, 8)
    probe_ep = args.probe_epochs or default_epochs(args.quick, 4)
    bs = 128 if args.dataset == "mnist" else 256

    tv = twoview_dm(args.dataset, args.quick, 256, labeled=False, data_dir=args.data_dir, num_workers=args.num_workers, max_files_per_class=args.max_files_per_class)
    module = SIGRegSSLModule(make_encoder(args.dataset), projector=make_projector(args.dataset))
    fit_or_load(module, tv, ssl_ep, args.quick, ckpt=args.ckpt)
    backbone = frozen_encoder(module)

    nc = n_classes(args.dataset)
    dm = plain_dm(args.dataset, args.quick, bs, data_dir=args.data_dir, num_workers=args.num_workers, max_files_per_class=args.max_files_per_class); dm.setup()
    head = nn.Linear(EMB_DIM, nc).to(DEVICE)
    train_linear_probe(backbone, head, dm.train_dataloader(), probe_ep)
    probs, labels = collect_probs(lambda x: head(backbone(x)), dm.test_dataloader())
    plot_roc(probs, labels, f"SIGReg (SSL) embedding + frozen linear head ROC [{args.dataset}]",
             plot_path(outfile(args.dataset, "roc_sigreg_linear.png")), n_classes=nc)

    embs, elab = collect_embeddings(backbone, dm.test_dataloader())
    plot_corner(embs, elab, plot_path(outfile(args.dataset, "corner_sigreg_16d.png")),
                title=f"SIGReg 16-dim latent space [{args.dataset}]")


if __name__ == "__main__":
    main()
