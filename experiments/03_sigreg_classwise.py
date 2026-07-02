"""
Class-conditional SIGReg with a chosen mean-geometry strategy, frozen linear probe.

Works on either dataset via --dataset {mnist,jetclass}.

--mode fixed       fixed orthogonal anchors (means not trained)
--mode learnmeans  learnable means + hinge separation term
--mode repulse     learnable means + inverse-square repulsion + shrinkage

Canonical training, e.g.:
    python cli.py fit --config configs/mnist_sigreg_classwise_repulse.yaml
    python cli.py fit --config configs/jetclass_sigreg_classwise_repulse.yaml
"""
import argparse
import numpy as np
import torch
import torch.nn as nn

from common import (default_epochs, fit_or_load, frozen_encoder,
                    make_encoder, plain_dm, classwise_dm, outfile, n_classes, DATASETS)
from models.config import plot_path, EMB_DIM, DEVICE
from models.litmodels import ClasswiseSIGRegModule
from utils.eval import train_linear_probe, collect_probs, collect_embeddings
from utils.plotting import plot_roc, plot_corner

TITLES = {
    "fixed": "fixed anchors",
    "learnmeans": "learnable means (hinge)",
    "repulse": "repulsive means",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="mnist")
    ap.add_argument("--mode", choices=list(TITLES), default="fixed")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ssl-epochs", type=int, default=None)
    ap.add_argument("--probe-epochs", type=int, default=None)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--data-dir", type=str, default=None,
                    help="JetClass ROOT directory (falls back to $JETCLASS_DIR / toy)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ssl_ep = args.ssl_epochs or default_epochs(args.quick, 8)
    probe_ep = args.probe_epochs or default_epochs(args.quick, 4)
    bs = 256

    nc = n_classes(args.dataset)
    emb_dm = classwise_dm(args.dataset, args.quick, bs, data_dir=args.data_dir)
    module = ClasswiseSIGRegModule(make_encoder(args.dataset), mode=args.mode, n_classes=nc)
    fit_or_load(module, emb_dm, ssl_ep, args.quick, ckpt=args.ckpt)
    backbone = frozen_encoder(module)

    dm = plain_dm(args.dataset, args.quick, bs, data_dir=args.data_dir); dm.setup()
    head = nn.Linear(EMB_DIM, nc).to(DEVICE)
    train_linear_probe(backbone, head, dm.train_dataloader(), probe_ep)
    probs, labels = collect_probs(lambda x: head(backbone(x)), dm.test_dataloader())
    plot_roc(probs, labels,
             f"Class-conditional SIGReg ({TITLES[args.mode]}) + linear head ROC [{args.dataset}]",
             plot_path(outfile(args.dataset, f"roc_sigreg_{args.mode}_linear.png")), n_classes=nc)

    embs, elab = collect_embeddings(backbone, dm.test_dataloader())
    plot_corner(embs, elab, plot_path(outfile(args.dataset, f"corner_sigreg_{args.mode}_16d.png")),
                title=f"Class-conditional SIGReg ({TITLES[args.mode]}) 16-dim latent [{args.dataset}]")


if __name__ == "__main__":
    main()
