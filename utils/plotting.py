"""ROC and corner-plot helpers."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from models.config import N_CLASSES


def plot_roc(probs, labels, title, out_path, n_classes=None):
    """One-vs-rest ROC (per class + micro-average) for a multi-class classifier."""
    n_classes = n_classes or N_CLASSES
    y_bin = label_binarize(labels, classes=list(range(n_classes)))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(7, 7))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=1, alpha=0.7, label=f"digit {i} (AUC={roc_auc[i]:.3f})")
    plt.plot(fpr["micro"], tpr["micro"], "k--", lw=2.5,
             label=f"micro-avg (AUC={roc_auc['micro']:.3f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1, ls=":")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(title); plt.legend(loc="lower right", fontsize=8); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"  saved {out_path}  (micro-AUC={roc_auc['micro']:.4f})")
    return roc_auc["micro"]


def plot_binary_roc(scores, y_true, title, out_path, label="model"):
    """Single ROC curve for a binary (one-vs-rest) score; returns the AUC."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k:", lw=1)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(title); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"  saved {out_path}  (AUC={roc_auc:.4f})")
    return fpr, tpr, roc_auc


def plot_corner(embs, labels, out_path, title=None, max_per_class=400):
    """Corner plot of the latent space with points/contours colored by class label."""
    import corner

    d = embs.shape[1]
    classes = np.unique(labels)
    cmap = plt.get_cmap("tab10")
    lo, hi = embs.min(axis=0), embs.max(axis=0)
    pad = 0.05 * (hi - lo + 1e-9)
    rng = list(zip(lo - pad, hi + pad))
    lbls = [f"$z_{{{i}}}$" for i in range(d)]

    fig = None
    for k, c in enumerate(classes):
        z = embs[labels == c]
        if len(z) > max_per_class:
            z = z[np.random.choice(len(z), max_per_class, replace=False)]
        fig = corner.corner(
            z, fig=fig, color=cmap(k % 10), bins=30, range=rng, labels=lbls,
            plot_datapoints=True, plot_density=False, plot_contours=True,
            fill_contours=False, hist_kwargs={"density": True},
            data_kwargs={"alpha": 0.35, "ms": 1.5}, contour_kwargs={"linewidths": 0.6},
        )
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=cmap(k % 10),
                          label=str(int(c))) for k, c in enumerate(classes)]
    fig.legend(handles=handles, loc="upper right", title="class", fontsize=9)
    if title:
        fig.suptitle(title, y=1.0)
    fig.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out_path}")


# --------------------------------------------------------------------------- #
# Reference-style corner plot (phlab-neurips25 utils.plotting.make_corner)      #
#                                                                             #
# Faithful port of the reference's embedding plot -- the same one it logs to    #
# Weights & Biases: an N x N grid of plain-matplotlib axes with step histograms #
# on the diagonal and s=0.5 scatter in the lower triangle, one default          #
# color-cycle colour ("C0", "C1", ...) per class, and a Patch legend in the     #
# top-right cell.  Kept byte-compatible in style; extended only with optional   #
# title / save so it can be used stand-alone as well as returned as a figure.   #
# --------------------------------------------------------------------------- #
def make_corner(x, labels, label_names=None, axwidth=2, return_fig=False):
    N = x.shape[1]
    fig, axes = plt.subplots(N, N, figsize=(N * axwidth, N * axwidth))
    for i in range(N):
        for j in range(N):
            plt.sca(axes[i, j])
            plt.axis("off")

    unique_labels = sorted(list(set(labels)))
    patches = []
    xlims = [[np.quantile(x[:, i], 0.00), np.quantile(x[:, i], 1.0)] for i in range(N)]
    bins = [np.linspace(xlims[i][0], xlims[i][1], 20) for i in range(N)]
    for il, label in enumerate(unique_labels):
        mask = labels == label
        for i in range(N):
            plt.sca(axes[i, i])
            plt.axis("on")
            plt.hist(x[mask, i], bins=bins[i], density=True, histtype="step", color=f"C{il}")

        for i in range(1, N):
            for j in range(i):
                plt.sca(axes[i, j])
                plt.scatter(x[mask, j], x[mask, i], s=0.5, color=f"C{il}")
                plt.xlim(axes[j, j].get_xlim())

        patches.append(Patch(label=label_names[label] if label_names is not None else label,
                             color=f"C{il}"))

    plt.sca(axes[0, -1])
    plt.legend(handles=patches, ncol=3)
    if return_fig:
        return fig


def save_corner(x, labels, out_path, label_names=None, title=None, axwidth=2, dpi=150):
    """Draw the reference-style ``make_corner`` and save it to ``out_path``."""
    fig = make_corner(x, labels, label_names=label_names, axwidth=axwidth, return_fig=True)
    if title:
        fig.suptitle(title)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out_path}")
