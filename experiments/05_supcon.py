"""
Supervised SimCLR (SupCon) embedding: closed-set (10-way) and hold-out studies.

Works on either dataset via --dataset {mnist,jetclass}.  Runs two experiments:
    (1) no holdout -> 10-way linear probe, ROC + corner
    (2) holdout    -> held-out-vs-rest binary probe, ROC + corner

Canonical training:
    python cli.py fit --config configs/mnist_supcon.yaml
    python cli.py fit --config configs/jetclass_supcon.yaml
"""
import argparse
import numpy as np
import torch
import torch.nn as nn

from common import (default_epochs, fit_or_load, frozen_encoder,
                    make_encoder, plain_dm, twoview_dm, outfile, CLASS_WORD, DATASETS)
from models.config import plot_path, EMB_DIM, N_CLASSES, HOLDOUT, DEVICE
from models.litmodels import SupConModule
from utils.eval import (
    train_linear_probe, train_binary_probe,
    collect_probs, collect_binary_scores, collect_embeddings,
)
from utils.plotting import plot_roc, plot_binary_roc, plot_corner


def run_no_holdout(dataset, quick, ssl_ep, probe_ep):
    print(f"\n===== SupCon, NO holdout (10-way) [{dataset}] =====")
    tv = twoview_dm(dataset, quick, 256, labeled=True)
    module = SupConModule(make_encoder(dataset))
    fit_or_load(module, tv, ssl_ep, quick)
    backbone = frozen_encoder(module)

    dm = plain_dm(dataset, quick, 256); dm.setup()
    head = nn.Linear(EMB_DIM, N_CLASSES).to(DEVICE)
    train_linear_probe(backbone, head, dm.train_dataloader(), probe_ep)
    probs, labels = collect_probs(lambda x: head(backbone(x)), dm.test_dataloader())
    plot_roc(probs, labels, f"Supervised SimCLR (SupCon) + linear head ROC [{dataset}]",
             plot_path(outfile(dataset, "roc_supcon_linear.png")))
    embs, elab = collect_embeddings(backbone, dm.test_dataloader())
    plot_corner(embs, elab, plot_path(outfile(dataset, "corner_supcon_16d.png")),
                title=f"SupCon 16-dim latent space [{dataset}]")


def run_holdout(dataset, quick, ssl_ep, probe_ep):
    word = CLASS_WORD[dataset]
    print(f"\n===== SupCon, HOLDOUT {HOLDOUT} ({HOLDOUT}-vs-rest) [{dataset}] =====")
    tv = twoview_dm(dataset, quick, 256, labeled=True, holdout=HOLDOUT)
    module = SupConModule(make_encoder(dataset))
    fit_or_load(module, tv, ssl_ep, quick)
    backbone = frozen_encoder(module)

    dm = plain_dm(dataset, quick, 256); dm.setup()
    head = nn.Linear(EMB_DIM, 2).to(DEVICE)
    train_binary_probe(backbone, head, dm.train_dataloader(), probe_ep)
    scores, ytrue = collect_binary_scores(backbone, head, dm.test_dataloader())
    plot_binary_roc(scores, ytrue, f"SupCon hold-out-{HOLDOUT} detection ROC [{dataset}]",
                    plot_path(outfile(dataset, "roc_supcon_holdout4.png")), label="SupCon")
    embs, elab = collect_embeddings(backbone, dm.test_dataloader())
    is_h = (elab == HOLDOUT).astype(int)
    plot_corner(embs, is_h, plot_path(outfile(dataset, "corner_supcon_holdout4.png")),
                title=f"SupCon hold-out-{HOLDOUT} latent [{dataset}]: "
                      f"1={word} {HOLDOUT} (unseen), 0=rest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="mnist")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ssl-epochs", type=int, default=None)
    ap.add_argument("--probe-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ssl_ep = args.ssl_epochs or default_epochs(args.quick, 8)
    probe_ep = args.probe_epochs or default_epochs(args.quick, 4)

    run_no_holdout(args.dataset, args.quick, ssl_ep, probe_ep)
    run_holdout(args.dataset, args.quick, ssl_ep, probe_ep)
    print("\nDone.")


if __name__ == "__main__":
    main()
