"""Self-supervised SIGReg (invariance + global isotropic Gaussian), frozen linear probe.

Trains via the Lightning `SIGRegSSLModule` (or loads --ckpt), freezes the
encoder, fits a linear probe and draws ROC + corner plots.  Canonical training:
    python cli.py fit --config configs/mnist_sigreg_ssl.yaml
"""
import argparse
import numpy as np
import torch
import torch.nn as nn

from common import default_epochs, fit_or_load, frozen_encoder
from models.config import plot_path, EMB_DIM, N_CLASSES, DEVICE
from models.networks import ConvBackbone
from models.litmodels import SIGRegSSLModule
from data.datasets import MNISTDataModule, TwoViewMNISTDataModule
from utils.eval import train_linear_probe, collect_probs, collect_embeddings
from utils.plotting import plot_roc, plot_corner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ssl-epochs", type=int, default=None)
    ap.add_argument("--probe-epochs", type=int, default=None)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ssl_ep = args.ssl_epochs or default_epochs(args.quick, 8)
    probe_ep = args.probe_epochs or default_epochs(args.quick, 4)

    tv = TwoViewMNISTDataModule(quick=args.quick, labeled=False, batch_size=256)
    module = SIGRegSSLModule(ConvBackbone())
    fit_or_load(module, tv, ssl_ep, args.quick, ckpt=args.ckpt)
    backbone = frozen_encoder(module)

    dm = MNISTDataModule(quick=args.quick, batch_size=128); dm.setup()
    head = nn.Linear(EMB_DIM, N_CLASSES).to(DEVICE)
    train_linear_probe(backbone, head, dm.train_dataloader(), probe_ep)
    probs, labels = collect_probs(lambda x: head(backbone(x)), dm.test_dataloader())
    plot_roc(probs, labels, "SIGReg (SSL) embedding + frozen linear head ROC",
             plot_path("roc_sigreg_linear.png"))

    embs, elab = collect_embeddings(backbone, dm.test_dataloader())
    plot_corner(embs, elab, plot_path("corner_sigreg_16d.png"),
                title="SIGReg 16-dim latent space (colored by digit)")


if __name__ == "__main__":
    main()
