"""
JetClass data: the **vendored weaver** dataloading stack + a thin adapter.

The reference repo (phlab-neurips25) loads JetClass with weaver's streaming
``SimpleIterDataset`` driven by a YAML data config.  That whole stack is vendored
verbatim under ``data/jetclass/`` (``dataset.py``, ``fileio.py``, ``preprocess.py``,
``config.py``, ``tools.py``) and the config under
``configs/jetclass_data_configs/JetClass_full.yaml`` -- so feature building,
manual standardization, wrap-padding, selection, and (optional) reweighting are
exactly the reference's.

``SimpleIterDataset`` yields per-jet ``(X, y, Z)`` where ``X`` is a dict of input
groups (``pf_points``/``pf_features``/``pf_vectors``/``pf_mask``, each ``[C, P]``),
``y['_label_']`` is the class index, and ``Z`` are observer variables (test only).
``JetClassAdapter`` converts that into the tensors our (SIGReg / SupCon) Lightning
modules expect, packing::

    x = [ pf_features (17) | pf_vectors (4) | pf_mask (1) ]  ->  [P, 22]

with the mask kept as the last channel (wrap-padding means padded slots are *not*
zero, so ParT reads the explicit mask).  Two augmented views (for SIGReg-SSL /
SupCon) are produced here via ``JetAugment``.
"""
import glob
import os

import numpy as np
import torch

from .jetclass.dataset import SimpleIterDataset

# Ten JetClass categories (canonical label-branch order).
JETCLASS_CLASSES = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]

# Five classes used for embedding training (matches the vendored data config's
# `labels.value` order -> contiguous labels 0..4).
DEFAULT_CLASSES = ["QCD", "Tbqq", "Wqq", "Zqq", "Hbb"]

# JetClass files are single-class, named by these prefixes.
CLASS_FILE_HEADERS = {
    "QCD": "ZJetsToNuNu", "Hbb": "HToBB", "Hcc": "HToCC", "Hgg": "HToGG",
    "H4q": "HToWW4Q", "Hqql": "HToWW2Q1L", "Zqq": "ZToQQ", "Wqq": "WToQQ",
    "Tbqq": "TTBar", "Tbl": "TTBarLep",
}

# Split -> subdirectory on the reference cluster.
SPLIT_DIRS = {"train": "train_100M", "val": "val_5M", "test": "test_20M"}

N_PARTICLES = 64            # `length` of every input group in the data config

# Packed-channel layout produced by the adapter (features + vectors + mask).
N_FEATURES = 17            # pf_features
N_VECTORS = 4              # pf_vectors (px, py, pz, energy)
N_CHANNELS = N_FEATURES + N_VECTORS + 1   # 22 (+ mask)
I_DR, I_DETA, I_DPHI = 4, 15, 16          # within pf_features (data-config order)
I_PX, I_PY = N_FEATURES, N_FEATURES + 1   # within pf_vectors
I_MASK = N_FEATURES + N_VECTORS           # 21


# --------------------------------------------------------------------------- #
# File discovery                                                              #
# --------------------------------------------------------------------------- #
def _split_dir(data_dir, split):
    """Resolve the directory for a split (train_100M / plain split/ / data_dir)."""
    if data_dir is None:
        return None
    for sub in (SPLIT_DIRS.get(split, split), split, ""):
        cand = os.path.join(data_dir, sub) if sub else data_dir
        if os.path.isdir(cand) and glob.glob(os.path.join(cand, "**", "*.root"), recursive=True):
            return cand
    return None


def build_file_dict(data_dir, split, class_names, max_files_per_class=None):
    """
    ``{class_name: [root files]}`` for ``split`` (the ``file_dict`` weaver wants).
    Returns ``None`` if no files are found (the DataModule turns that into an error).
    """
    split_dir = _split_dir(data_dir, split)
    if split_dir is None:
        return None
    out = {}
    for c in class_names:
        header = CLASS_FILE_HEADERS[c]
        files = sorted(glob.glob(os.path.join(split_dir, "**", f"{header}_*.root"),
                                 recursive=True))
        if max_files_per_class:
            files = files[:max_files_per_class]
        if files:
            out[c] = files
    return out or None


def make_iter_dataset(file_dict, data_config, for_training, fetch_step=0.01,
                      file_fraction=1, load_fraction=1, name=""):
    """Construct a weaver ``SimpleIterDataset`` (streaming, bounded memory)."""
    return SimpleIterDataset(
        file_dict, data_config,
        for_training=for_training,
        load_range_and_fraction=((0, 1), load_fraction),
        fetch_by_files=False,
        fetch_step=fetch_step,
        file_fraction=file_fraction,
        remake_weights=False,
        async_load=False,
        infinity_mode=False,
        in_memory=False,
        name=name,
    )


# --------------------------------------------------------------------------- #
# Augmentation (two-view) + adapter to our tensor format                       #
# --------------------------------------------------------------------------- #
class JetAugment:
    """
    Random eta-phi rotation of the relative coordinates + mild pt / log smearing.
    The same azimuthal rotation is applied to (px, py) so ParT's rotation-invariant
    pairwise features stay consistent.  Operates on the packed ``[P, 22]`` tensor
    and leaves the mask channel untouched (padding is handled by the mask, not by
    zeroing -- the data config wrap-pads).
    """

    def __init__(self, ang_sigma=0.02, logpt_sigma=0.05):
        self.ang_sigma = ang_sigma
        self.logpt_sigma = logpt_sigma

    def __call__(self, feats):
        feats = feats.clone()
        theta = torch.rand(1, device=feats.device) * 2 * np.pi
        cos, sin = torch.cos(theta), torch.sin(theta)

        deta, dphi = feats[:, I_DETA].clone(), feats[:, I_DPHI].clone()
        feats[:, I_DETA] = cos * deta - sin * dphi
        feats[:, I_DPHI] = sin * deta + cos * dphi          # deltaR (I_DR) preserved

        px, py = feats[:, I_PX].clone(), feats[:, I_PY].clone()
        feats[:, I_PX] = cos * px - sin * py                # azimuthal rotation of the
        feats[:, I_PY] = sin * px + cos * py                # 4-vector (pairwise-invariant)

        feats[:, :4] += torch.randn_like(feats[:, :4]) * self.logpt_sigma
        feats[:, I_DETA] += torch.randn_like(feats[:, I_DETA]) * self.ang_sigma
        feats[:, I_DPHI] += torch.randn_like(feats[:, I_DPHI]) * self.ang_sigma
        return feats                                        # mask channel left as-is


class JetClassAdapter(torch.utils.data.IterableDataset):
    """
    Wrap a weaver ``SimpleIterDataset`` and yield our per-sample formats:

        mode="plain"           -> (features[P,22], label)
        mode="twoview"         -> (view1, view2)
        mode="twoview_labeled" -> (view1, view2, label)
    """

    def __init__(self, iter_ds, mode="plain", aug=None):
        super().__init__()
        self.iter_ds = iter_ds
        self.mode = mode
        self.aug = aug or JetAugment()

    def __iter__(self):
        for X, y, _ in self.iter_ds:                        # weaver yields (X, y, Z)
            packed = np.concatenate(
                [X["pf_features"], X["pf_vectors"], X["pf_mask"]], axis=0)  # [22, P]
            xt = torch.from_numpy(np.ascontiguousarray(packed.T)).float()  # [P, 22]
            label = int(np.asarray(y["_label_"]).reshape(-1)[0])
            if self.mode == "plain":
                yield xt, label
            elif self.mode == "twoview":
                yield self.aug(xt), self.aug(xt)
            else:                                           # twoview_labeled
                yield self.aug(xt), self.aug(xt), label
