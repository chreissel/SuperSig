"""
JetClass dataset primitives.

JetClass (Qu, Li & Qian, 2022, https://arxiv.org/abs/2202.03772) is a jet-tagging
benchmark with ten jet classes.  Mirroring the reference repo (phlab-neurips25),
the embedding is trained on a **five-class** subset by default -- QCD, Tbqq
(ttbar), Wqq, Zqq, Hbb (see ``DEFAULT_CLASSES``).  All ten can be selected by
passing an explicit ``classes`` list.

Each jet is a fixed-size particle cloud with, per particle, the reference's full
**17 input features** followed by the four-momentum **vectors** used by
ParticleTransformer for its pairwise interaction features::

    features (17): log_pt, log_e, log_ptrel, log_erel, deltaR,
                   charge, isChargedHadron, isNeutralHadron, isPhoton,
                   isElectron, isMuon, d0, d0err, dz, dzerr, deta, dphi
    vectors  (4):  px, py, pz, energy

so every jet tensor has ``N_CHANNELS = 21`` channels.  The 17 features are
standardized with the same manual center/scale/clip values as the reference data
config (``JetClass_full.yaml``); the four vectors are kept raw (ParT derives its
own pairwise quantities from them).  Padded slots are all-zero, the sentinel the
encoders use to build the particle mask.

Two data paths are provided:

* **real** -- read the official JetClass ROOT files with ``uproot`` (lazy import),
  from ``JETCLASS_DIR`` (or a directory passed to the DataModule).  PID / impact-
  parameter branches are used when present.
* **toy**  -- a self-contained synthetic generator (class-dependent prong
  structure), so the *exact same* test suite runs anywhere without the ~100 GB
  download.  If the real files are not found the DataModules fall back to this.

The two-view augmentation (SIGReg-SSL / SupCon) is an eta-phi rotation of the
relative coordinates plus mild pt smearing; the same azimuthal rotation is applied
to (px, py) so ParT's rotation-invariant pairwise features stay consistent.  As in
the reference, the augmentation lives here in the loader, not in the module.
"""
import glob
import os

import numpy as np
import torch

# Ten JetClass categories, in the canonical order used by the official label
# branches (``label_QCD`` ... ``label_Tbl``).
JETCLASS_CLASSES = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]
LABEL_BRANCHES = [f"label_{c}" for c in JETCLASS_CLASSES]

# The five classes used for embedding training in the reference repo
# (phlab-neurips25, configs/jetclass_data_configs/JetClass_full.yaml:
#  value: [label_QCD, label_Tbqq, label_Wqq, label_Zqq, label_Hbb]).
DEFAULT_CLASSES = ["QCD", "Tbqq", "Wqq", "Zqq", "Hbb"]

P4_BRANCHES = ["part_px", "part_py", "part_pz", "part_energy"]
# Optional per-particle branches (PID flags + impact parameters); used when present.
EXTRA_BRANCHES = ["part_charge", "part_isChargedHadron", "part_isNeutralHadron",
                  "part_isPhoton", "part_isElectron", "part_isMuon",
                  "part_d0val", "part_d0err", "part_dzval", "part_dzerr"]

# Feature layout (17), matching JetClass_full.yaml `pf_features`, then 4 vectors.
FEATURE_NAMES = ["log_pt", "log_e", "log_ptrel", "log_erel", "deltaR",
                 "charge", "isChargedHadron", "isNeutralHadron", "isPhoton",
                 "isElectron", "isMuon", "d0", "d0err", "dz", "dzerr", "deta", "dphi"]
N_FEATURES = len(FEATURE_NAMES)          # 17
N_VECTORS = len(P4_BRANCHES)             # 4  (px, py, pz, energy)
N_CHANNELS = N_FEATURES + N_VECTORS      # 21

I_DR, I_DETA, I_DPHI = 4, 15, 16         # within the feature block
I_PX, I_PY = N_FEATURES + 0, N_FEATURES + 1   # within the vector block

N_PARTICLES = 64          # particles per jet after padding / truncation

# Per-feature standardization (center, scale, clip_min, clip_max) from the
# reference data config; ``None`` means no transform (raw value kept).
STD = [
    (1.7, 0.7, -5, 5),    # log_pt
    (2.0, 0.7, -5, 5),    # log_e
    (-4.7, 0.7, -5, 5),   # log_ptrel
    (-4.7, 0.7, -5, 5),   # log_erel
    (0.2, 4.0, -5, 5),    # deltaR
    None,                 # charge
    None,                 # isChargedHadron
    None,                 # isNeutralHadron
    None,                 # isPhoton
    None,                 # isElectron
    None,                 # isMuon
    None,                 # d0 (= tanh(d0val))
    (0, 1, 0, 1),         # d0err
    None,                 # dz (= tanh(dzval))
    (0, 1, 0, 1),         # dzerr
    None,                 # deta
    None,                 # dphi
]


# --------------------------------------------------------------------------- #
# Feature computation (shared by the real and toy paths)                       #
# --------------------------------------------------------------------------- #
def compute_features(px, py, pz, energy, charge=None, isChargedHadron=None,
                     isNeutralHadron=None, isPhoton=None, isElectron=None,
                     isMuon=None, d0val=None, d0err=None, dzval=None, dzerr=None):
    """
    Build the ``[N, P, 21]`` jet tensor (17 standardized features + 4 raw vectors)
    from padded per-particle arrays (each ``[N, P]``, zero-padded).  Optional
    PID / impact-parameter arrays default to zero when absent.
    """
    px, py, pz, energy = (np.asarray(a, dtype=np.float64) for a in (px, py, pz, energy))
    n, p = px.shape
    z = np.zeros((n, p))

    def opt(a, fn=None):
        if a is None:
            return z.copy()
        a = np.asarray(a, dtype=np.float64)
        return fn(a) if fn else a

    pt = np.hypot(px, py)
    mask = pt > 0                                            # real particles
    eta = np.arcsinh(np.divide(pz, pt, out=np.zeros_like(pz), where=mask))
    phi = np.arctan2(py, px)

    jpx, jpy, jpz, je = (a.sum(axis=1) for a in (px, py, pz, energy))
    jpt = np.hypot(jpx, jpy)[:, None]
    jeta = np.arcsinh(np.divide(jpz, jpt[:, 0], out=np.zeros_like(jpz), where=jpt[:, 0] > 0))[:, None]
    jphi = np.arctan2(jpy, jpx)[:, None]
    je = je[:, None]

    deta = eta - jeta
    dphi = np.arctan2(np.sin(phi - jphi), np.cos(phi - jphi))
    dR = np.hypot(deta, dphi)

    eps = 1e-8
    log_pt = np.log(pt + eps)
    log_e = np.log(energy + eps)
    log_ptrel = np.log(pt / (jpt + eps) + eps)
    log_erel = np.log(energy / (je + eps) + eps)

    feats = np.stack([
        log_pt, log_e, log_ptrel, log_erel, dR,
        opt(charge), opt(isChargedHadron), opt(isNeutralHadron), opt(isPhoton),
        opt(isElectron), opt(isMuon),
        opt(d0val, np.tanh), opt(d0err), opt(dzval, np.tanh), opt(dzerr),
        deta, dphi,
    ], axis=-1)                                              # [N, P, 17]

    for i, params in enumerate(STD):
        if params is not None:
            c, s, lo, hi = params
            feats[..., i] = np.clip((feats[..., i] - c) * s, lo, hi)

    vectors = np.stack([px, py, pz, energy], axis=-1)        # [N, P, 4] (raw)
    x = np.concatenate([feats, vectors], axis=-1)            # [N, P, 21]
    x = x * mask[..., None]                                  # zero the padded slots
    return x.astype(np.float32)


# --------------------------------------------------------------------------- #
# Synthetic ("toy") jets                                                        #
# --------------------------------------------------------------------------- #
def gen_toy_jets(n_per_class, class_indices, n_particles=N_PARTICLES, seed=None):
    """
    Class-separable synthetic jets.  Class ``c`` gets ``1 + c % 4`` prongs and a
    class-dependent pt scale, so a linear probe on a learned embedding is
    meaningful (ROC > 0.5) without the real dataset.  PID / impact-parameter
    features are left at zero (the class signal is kinematic).
    """
    rng = np.random.default_rng(seed)
    px, py, pz, en, ys = [], [], [], [], []
    for c in class_indices:
        n_sub = 1 + (c % 4)
        pt_scale = 1.0 + 0.15 * c
        for _ in range(n_per_class):
            npart = int(rng.integers(20, n_particles))
            eta0, phi0 = rng.normal(0, 1.0), rng.uniform(-np.pi, np.pi)
            centers = rng.uniform(-0.5, 0.5, size=(n_sub, 2))
            assign = rng.integers(0, n_sub, size=npart)
            spread = 0.08 + 0.04 * (c % 3)
            deta = centers[assign, 0] + rng.normal(0, spread, npart)
            dphi = centers[assign, 1] + rng.normal(0, spread, npart)
            pt = pt_scale * rng.exponential(3.0, npart) + 0.5
            eta, phi = eta0 + deta, phi0 + dphi
            row_px = pt * np.cos(phi)
            row_py = pt * np.sin(phi)
            row_pz = pt * np.sinh(eta)
            row_e = pt * np.cosh(eta)
            pad = n_particles - npart
            px.append(np.pad(row_px, (0, pad)))
            py.append(np.pad(row_py, (0, pad)))
            pz.append(np.pad(row_pz, (0, pad)))
            en.append(np.pad(row_e, (0, pad)))
            ys.append(c)
    X = compute_features(np.array(px), np.array(py), np.array(pz), np.array(en))
    y = np.array(ys, dtype=np.int64)
    perm = rng.permutation(len(y))
    return torch.from_numpy(X[perm]), torch.from_numpy(y[perm])


# --------------------------------------------------------------------------- #
# Real JetClass ROOT files (uproot, lazily imported)                           #
# --------------------------------------------------------------------------- #
def _find_files(data_dir, split):
    if data_dir is None:
        return []
    for cand in (os.path.join(data_dir, split), data_dir):
        files = sorted(glob.glob(os.path.join(cand, "**", "*.root"), recursive=True))
        if files:
            return files
    return []


def load_root(data_dir, split, class_indices, n_particles=N_PARTICLES,
              max_events=None, max_files=None):
    """
    Load real JetClass jets for ``split`` (``train`` / ``val`` / ``test``).

    Returns ``(X, y)`` tensors, or ``None`` if no ROOT files are found (so callers
    can fall back to the toy generator).
    """
    files = _find_files(data_dir, split)
    if not files:
        return None
    if max_files:
        files = files[:max_files]

    import uproot          # lazy: only needed for the real dataset
    import awkward as ak

    keep = set(int(c) for c in class_indices)
    label_cols = [LABEL_BRANCHES[c] for c in class_indices]
    remap = {c: i for i, c in enumerate(class_indices)}   # original -> contiguous

    Xs, ys, n_seen = [], [], 0
    for path in files:
        f = uproot.open(path)
        tree = f["tree"] if "tree" in f else f[f.keys(recursive=False)[0]]
        avail = set(tree.keys())
        want = P4_BRANCHES + [b for b in EXTRA_BRANCHES if b in avail]
        arr = tree.arrays(want + label_cols, library="ak")

        onehot = np.stack([ak.to_numpy(arr[b]).astype(np.int64) for b in label_cols], axis=1)
        cls_local = onehot.argmax(axis=1)
        has_label = onehot.sum(axis=1) > 0
        cls_orig = np.array([class_indices[i] for i in cls_local])
        sel = has_label & np.array([int(c) in keep for c in cls_orig])
        if not sel.any():
            continue

        def pad(branch):
            a = arr[branch][sel]
            a = ak.pad_none(a, n_particles, clip=True)
            return ak.to_numpy(ak.fill_none(a, 0.0)).astype(np.float64)

        # map branch names -> compute_features kwargs
        kwargs = {}
        alias = {"part_charge": "charge", "part_isChargedHadron": "isChargedHadron",
                 "part_isNeutralHadron": "isNeutralHadron", "part_isPhoton": "isPhoton",
                 "part_isElectron": "isElectron", "part_isMuon": "isMuon",
                 "part_d0val": "d0val", "part_d0err": "d0err",
                 "part_dzval": "dzval", "part_dzerr": "dzerr"}
        for b in EXTRA_BRANCHES:
            if b in avail:
                kwargs[alias[b]] = pad(b)

        X = compute_features(pad("part_px"), pad("part_py"), pad("part_pz"),
                             pad("part_energy"), **kwargs)
        y = np.array([remap[int(c)] for c in cls_orig[sel]], dtype=np.int64)
        Xs.append(X); ys.append(y); n_seen += len(y)
        if max_events and n_seen >= max_events:
            break

    if not Xs:
        return None
    X = np.concatenate(Xs)[:max_events] if max_events else np.concatenate(Xs)
    y = np.concatenate(ys)[:max_events] if max_events else np.concatenate(ys)
    return torch.from_numpy(X), torch.from_numpy(y)


# --------------------------------------------------------------------------- #
# Augmentation + two-view datasets                                             #
# --------------------------------------------------------------------------- #
class JetAugment:
    """
    Random eta-phi rotation of the relative coordinates + mild pt / log smearing.
    The same azimuthal rotation is applied to (px, py) so ParT's rotation-invariant
    pairwise features stay consistent with the rotated node features.
    """

    def __init__(self, ang_sigma=0.02, logpt_sigma=0.05):
        self.ang_sigma = ang_sigma
        self.logpt_sigma = logpt_sigma

    def __call__(self, feats):
        feats = feats.clone()
        mask = (feats.abs().sum(-1, keepdim=True) > 0).float()

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
        return feats * mask                                 # keep padded slots zero


class JetDataset(torch.utils.data.Dataset):
    """Plain ``(features, label)`` jets."""

    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], int(self.y[i])


class TwoViewJets(torch.utils.data.Dataset):
    """Two augmented views of each jet (no label)."""

    def __init__(self, X, aug=None):
        self.X, self.aug = X, aug or JetAugment()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.aug(self.X[i]), self.aug(self.X[i])


class TwoViewLabeledJets(torch.utils.data.Dataset):
    """Two augmented views of each jet plus its label (for SupCon)."""

    def __init__(self, X, y, aug=None):
        self.X, self.y, self.aug = X, y, aug or JetAugment()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.aug(self.X[i]), self.aug(self.X[i]), int(self.y[i])
