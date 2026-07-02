"""Baseline: backbone + classifier trained end-to-end with cross-entropy + ROC.

Works on either dataset via --dataset {mnist,jetclass}.  Trains via the Lightning
`SupervisedModule` (or loads --ckpt), then evaluates on the test set.  Canonical:
    python cli.py fit --config configs/mnist_supervised.yaml
    python cli.py fit --config configs/jetclass_supervised.yaml
"""
import argparse
import numpy as np
import torch

from common import (default_epochs, fit_or_load, make_supervised_net, plain_dm,
                    outfile, n_classes, DATASETS)
from models.config import plot_path, DEVICE
from models.litmodels import SupervisedModule
from utils.eval import collect_probs
from utils.plotting import plot_roc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="mnist")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--data-dir", type=str, default=None,
                    help="JetClass ROOT directory (falls back to $JETCLASS_DIR / toy)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    epochs = args.epochs or default_epochs(args.quick, 3)
    bs = 128 if args.dataset == "mnist" else 256

    dm = plain_dm(args.dataset, args.quick, bs, data_dir=args.data_dir)
    module = SupervisedModule(make_supervised_net(args.dataset))
    fit_or_load(module, dm, epochs, args.quick, ckpt=args.ckpt)

    dm.setup()
    model = module.model.to(DEVICE).eval()
    probs, labels = collect_probs(lambda x: model(x), dm.test_dataloader())
    plot_roc(probs, labels, f"{args.dataset} supervised ROC",
             plot_path(outfile(args.dataset, "roc_supervised.png")), n_classes=n_classes(args.dataset))


if __name__ == "__main__":
    main()
