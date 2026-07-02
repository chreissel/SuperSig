"""
Hold-out study: a class-conditional SIGReg embedding trained WITHOUT one class,
frozen, then a binary "held-out vs rest" linear probe -- does the embedding place
the unseen class in its own region?

Works on either dataset via --dataset {mnist,jetclass} (the held-out label index
is `HOLDOUT`, i.e. digit 4 for MNIST / class 4 = H4q for JetClass).

--mode learnmeans | repulse | both

Canonical training, e.g.:
    python cli.py fit --config configs/mnist_sigreg_holdout4_repulse.yaml
    python cli.py fit --config configs/jetclass_sigreg_holdout4_repulse.yaml
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from common import (default_epochs, fit_or_load, frozen_encoder,
                    make_encoder, plain_dm, classwise_dm, outfile, CLASS_WORD, DATASETS)
from models.config import plot_path, EMB_DIM, HOLDOUT, DEVICE
from models.litmodels import ClasswiseSIGRegModule
from utils.eval import train_binary_probe, collect_binary_scores, collect_embeddings
from utils.plotting import plot_binary_roc, plot_corner


def run_mode(mode, dataset, probe_dm, ssl_ep, probe_ep, quick, ckpt=None):
    word = CLASS_WORD[dataset]
    print(f"\n===== MODE: {mode} (embedding trained WITHOUT {word} {HOLDOUT}) =====")
    emb_dm = classwise_dm(dataset, quick, 256, holdout=HOLDOUT)
    module = ClasswiseSIGRegModule(make_encoder(dataset), mode=mode)
    fit_or_load(module, emb_dm, ssl_ep, quick, ckpt=ckpt)
    backbone = frozen_encoder(module)

    print(f"  --- freeze, train {HOLDOUT}-vs-rest linear head ---")
    head = nn.Linear(EMB_DIM, 2).to(DEVICE)
    train_binary_probe(backbone, head, probe_dm.train_dataloader(), probe_ep)
    scores, ytrue = collect_binary_scores(backbone, head, probe_dm.test_dataloader())
    fpr, tpr, roc_auc = plot_binary_roc(
        scores, ytrue,
        f"Hold-out-{HOLDOUT} detection ROC ({mode}) [{dataset}]",
        plot_path(outfile(dataset, f"roc_holdout4_{mode}.png")), label=mode)

    embs, ylab = collect_embeddings(backbone, probe_dm.test_dataloader())
    is_h = (ylab == HOLDOUT).astype(int)
    plot_corner(embs, is_h, plot_path(outfile(dataset, f"corner_holdout4_{mode}.png")),
                title=f"Hold-out-{HOLDOUT} latent ({mode}) [{dataset}]: "
                      f"1={word} {HOLDOUT} (unseen), 0=rest")
    return fpr, tpr, roc_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="mnist")
    ap.add_argument("--mode", choices=["learnmeans", "repulse", "both"], default="both")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ssl-epochs", type=int, default=None)
    ap.add_argument("--probe-epochs", type=int, default=None)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ssl_ep = args.ssl_epochs or default_epochs(args.quick, 8)
    probe_ep = args.probe_epochs or default_epochs(args.quick, 4)

    # Probe/test on all classes (probe is trained with the held-out class present).
    probe_dm = plain_dm(args.dataset, args.quick, 256); probe_dm.setup()
    modes = ["repulse", "learnmeans"] if args.mode == "both" else [args.mode]
    results = {m: run_mode(m, args.dataset, probe_dm, ssl_ep, probe_ep, args.quick,
                           ckpt=args.ckpt if len(modes) == 1 else None)
               for m in modes}

    if len(results) > 1:
        plt.figure(figsize=(6, 6))
        for m, (fpr, tpr, a) in results.items():
            plt.plot(fpr, tpr, lw=2, label=f"{m} (AUC={a:.4f})")
        plt.plot([0, 1], [0, 1], "k:", lw=1)
        plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
        plt.title(f"Hold-out-{HOLDOUT} detection: repulsive vs learnable means [{args.dataset}]")
        plt.legend(loc="lower right"); plt.tight_layout()
        plt.savefig(plot_path(outfile(args.dataset, "roc_holdout4_compare.png")), dpi=150)
        plt.close()
        print(f"  saved {outfile(args.dataset, 'roc_holdout4_compare.png')}")


if __name__ == "__main__":
    main()
