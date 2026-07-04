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

The official JetClass ROOT files are read with ``uproot`` (lazy import) from
``JETCLASS_DIR`` (or a directory passed to the DataModule); PID / impact-parameter
branches are used when present.  Reading is **streaming** (``JetStream``, an
``IterableDataset``): files are read lazily in chunks, round-robin across classes,
with a bounded shuffle buffer -- the memory-safe analogue of the reference's weaver
``SimpleIterDataset``, so peak memory does not grow with the (up to ~100M-jet)
split size.

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
# Feature computation                                                          #
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
# JetClass ROOT files (uproot, lazily imported)                                #
# --------------------------------------------------------------------------- #
# Subdirectory holding each split on the reference cluster
# (/n/holystore01/LABS/iaifi_lab/Lab/sambt/JetClass/).
SPLIT_DIRS = {"train": "train_100M", "val": "val_5M", "test": "test_20M"}

# JetClass files are single-class, named by a per-class prefix; this lets us read
# a bounded, class-balanced subset instead of loading the whole (~100M-jet) split.
CLASS_FILE_HEADERS = {
    "QCD": "ZJetsToNuNu", "Hbb": "HToBB", "Hcc": "HToCC", "Hgg": "HToGG",
    "H4q": "HToWW4Q", "Hqql": "HToWW2Q1L", "Zqq": "ZToQQ", "Wqq": "WToQQ",
    "Tbqq": "TTBar", "Tbl": "TTBarLep",
}

_BRANCH_ALIAS = {
    "part_charge": "charge", "part_isChargedHadron": "isChargedHadron",
    "part_isNeutralHadron": "isNeutralHadron", "part_isPhoton": "isPhoton",
    "part_isElectron": "isElectron", "part_isMuon": "isMuon",
    "part_d0val": "d0val", "part_d0err": "d0err",
    "part_dzval": "dzval", "part_dzerr": "dzerr",
}


def _split_dir(data_dir, split):
    """Resolve the directory for a split (train_100M / plain split/ / data_dir)."""
    if data_dir is None:
        return None
    for sub in (SPLIT_DIRS.get(split, split), split, ""):
        cand = os.path.join(data_dir, sub) if sub else data_dir
        if os.path.isdir(cand) and glob.glob(os.path.join(cand, "**", "*.root"), recursive=True):
            return cand
    return None


def _arr_to_features(arr, n_particles):
    """Turn one awkward chunk (a record of jagged branches) into ``[n, P, 21]``."""
    import awkward as ak
    fields = set(arr.fields)

    def pad(branch):
        a = ak.pad_none(arr[branch], n_particles, clip=True)
        return ak.to_numpy(ak.fill_none(a, 0.0)).astype(np.float64)

    kwargs = {_BRANCH_ALIAS[b]: pad(b) for b in EXTRA_BRANCHES if b in fields}
    return compute_features(pad("part_px"), pad("part_py"), pad("part_pz"),
                            pad("part_energy"), **kwargs)


def class_file_dict(data_dir, split, class_indices, max_files_per_class=None):
    """
    Map each requested class (as a contiguous label 0..N-1) to its ROOT files for
    ``split``.  Cheap -- only globs filenames, reads nothing.  Returns ``None`` if
    no files are found (the DataModule turns that into a clear error).
    """
    split_dir = _split_dir(data_dir, split)
    if split_dir is None:
        return None
    out = {}
    for label, ci in enumerate(class_indices):
        header = CLASS_FILE_HEADERS[JETCLASS_CLASSES[ci]]
        files = sorted(glob.glob(os.path.join(split_dir, "**", f"{header}_*.root"),
                                 recursive=True))
        if max_files_per_class:
            files = files[:max_files_per_class]
        if files:
            out[label] = files
    return out or None


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


class JetStream(torch.utils.data.IterableDataset):
    """
    Streaming JetClass dataset (the memory-safe analogue of the reference's weaver
    ``SimpleIterDataset``).  ROOT files are read lazily in chunks with ``uproot``,
    round-robin across classes so batches stay class-balanced, and buffered/shuffled
    with a bounded shuffle buffer -- so peak memory is O(buffer), independent of the
    (up to ~100M-jet) split size.  Per-sample output matches the map-style loaders:

        mode="plain"           -> (features[P,21], label)
        mode="twoview"         -> (view1, view2)
        mode="twoview_labeled" -> (view1, view2, label)

    ``files_by_class`` maps a contiguous label to its list of ROOT files.
    """

    def __init__(self, files_by_class, n_particles=N_PARTICLES, mode="plain", aug=None,
                 chunk_size=10000, shuffle_buffer=20000, shuffle=True, seed=0):
        super().__init__()
        self.files_by_class = files_by_class
        self.n_particles = n_particles
        self.mode = mode
        self.aug = aug or JetAugment()
        self.chunk_size = chunk_size
        self.shuffle_buffer = shuffle_buffer
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def _class_chunks(self, files):
        import uproot          # lazy: only needed for the real dataset
        for path in files:
            f = uproot.open(path)
            tree = f["tree"] if "tree" in f else f[f.keys(recursive=False)[0]]
            avail = set(tree.keys())
            want = P4_BRANCHES + [b for b in EXTRA_BRANCHES if b in avail]
            for arr in tree.iterate(want, step_size=self.chunk_size, library="ak"):
                yield _arr_to_features(arr, self.n_particles)

    def _round_robin(self, file_rng):
        """Yield (label, chunk) round-robin across classes (shared order across workers)."""
        gens = {}
        for label, files in self.files_by_class.items():
            fl = list(files)
            if self.shuffle:
                file_rng.shuffle(fl)
            gens[label] = self._class_chunks(fl)
        active = dict(gens)
        while active:
            for label in list(active.keys()):
                try:
                    yield label, next(active[label])
                except StopIteration:
                    del active[label]

    def _emit(self, x, label):
        xt = torch.from_numpy(x)
        if self.mode == "plain":
            return xt, int(label)
        if self.mode == "twoview":
            return self.aug(xt), self.aug(xt)
        return self.aug(xt), self.aug(xt), int(label)     # twoview_labeled

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        wid = worker.id if worker else 0
        nworkers = worker.num_workers if worker else 1
        # Shared file ordering across workers (so chunk-index sharding is disjoint);
        # per-worker buffer shuffling.
        file_rng = np.random.default_rng(self.seed + 1000 * self._epoch)
        buf_rng = np.random.default_rng(self.seed + 1000 * self._epoch + 7 * wid)
        self._epoch += 1

        buffer, counter = [], 0
        for label, X in self._round_robin(file_rng):
            if counter % nworkers != wid:                 # chunk-level worker shard
                counter += 1
                continue
            counter += 1
            for i in range(len(X)):
                buffer.append((X[i], label))
            if len(buffer) >= self.shuffle_buffer:
                if self.shuffle:
                    buf_rng.shuffle(buffer)
                half = len(buffer) // 2
                out, buffer = buffer[:half], buffer[half:]
                for x, lab in out:
                    yield self._emit(x, lab)
        if self.shuffle:
            buf_rng.shuffle(buffer)
        for x, lab in buffer:
            yield self._emit(x, lab)
