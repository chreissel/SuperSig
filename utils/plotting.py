"""ROC and corner-plot helpers."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from models.config import N_CLASSES


def plot_roc(probs, labels, title, out_path):
    """One-vs-rest ROC (per class + micro-average) for a multi-class classifier."""
    y_bin = label_binarize(labels, classes=list(range(N_CLASSES)))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(N_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(7, 7))
    for i in range(N_CLASSES):
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
