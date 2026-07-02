"""Baseline: CNN trained end-to-end with categorical cross-entropy + ROC.

Trains via the Lightning `SupervisedModule` (or loads --ckpt), then evaluates
on the MNIST test set.  For the canonical training run see:
    python cli.py fit --config configs/mnist_supervised.yaml
"""
import argparse
import numpy as np
import torch

from common import default_epochs, fit_or_load
from models.config import plot_path, DEVICE
from models.networks import SupervisedCNN
from models.litmodels import SupervisedModule
from data.datasets import MNISTDataModule
from utils.eval import collect_probs
from utils.plotting import plot_roc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    epochs = args.epochs or default_epochs(args.quick, 3)

    dm = MNISTDataModule(quick=args.quick, batch_size=128)
    module = SupervisedModule(SupervisedCNN())
    fit_or_load(module, dm, epochs, args.quick, ckpt=args.ckpt)

    dm.setup()
    model = module.model.to(DEVICE).eval()
    probs, labels = collect_probs(lambda x: model(x), dm.test_dataloader())
    plot_roc(probs, labels, "MNIST supervised CNN ROC", plot_path("roc_supervised.png"))


if __name__ == "__main__":
    main()
