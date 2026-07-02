"""
Lightning DataModules for MNIST.

Three modules cover every experiment:

* ``MNISTDataModule``          -- plain ``(image, label)``, all classes.
* ``ClasswiseMNISTDataModule`` -- ``(image, label)`` with an optional held-out
  digit removed from the training split (class-conditional SIGReg / hold-out).
* ``TwoViewMNISTDataModule``   -- two augmented views per image, optionally
  labelled and/or with a held-out digit (SIGReg-SSL / SupCon).
"""
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import lightning as pl

from models.config import DATA_DIR, JETCLASS_DIR, HOLDOUT  # noqa: F401  (HOLDOUT is a handy default)
from . import data_utils as dutils
from . import jetclass_data as jc


class GenericDataModule(pl.LightningDataModule):
    """Common loader bookkeeping (batch size / workers / pin-memory)."""

    def __init__(self, batch_size=128, num_workers=2, pin_memory=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }


class MNISTDataModule(GenericDataModule):
    """Plain MNIST (all classes) yielding ``(image, label)``."""

    def __init__(self, quick=False, **kwargs):
        super().__init__(**kwargs)
        self.quick = quick

    def prepare_data(self):
        datasets.MNIST(DATA_DIR, train=True, download=True)
        datasets.MNIST(DATA_DIR, train=False, download=True)

    def setup(self, stage=None):
        train = datasets.MNIST(DATA_DIR, train=True, transform=dutils.TF_PLAIN)
        test = datasets.MNIST(DATA_DIR, train=False, transform=dutils.TF_PLAIN)
        if self.quick:
            train, test = Subset(train, range(4000)), Subset(test, range(2000))
        self.train_ds, self.test_ds = train, test

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.test_ds, shuffle=False, **self.loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_ds, shuffle=False, **self.loader_kwargs)


class ClasswiseMNISTDataModule(GenericDataModule):
    """MNIST ``(image, label)``; optionally drops ``holdout`` from training."""

    def __init__(self, quick=False, holdout=None, **kwargs):
        super().__init__(**kwargs)
        self.quick = quick
        self.holdout = holdout

    def prepare_data(self):
        datasets.MNIST(DATA_DIR, train=True, download=True)
        datasets.MNIST(DATA_DIR, train=False, download=True)

    def setup(self, stage=None):
        train = datasets.MNIST(DATA_DIR, train=True, transform=dutils.TF_PLAIN)
        test = datasets.MNIST(DATA_DIR, train=False, transform=dutils.TF_PLAIN)
        n = 8000 if self.quick else len(train)
        idx = [i for i in range(n)
               if (self.holdout is None or int(train.targets[i]) != self.holdout)]
        self.train_ds = Subset(train, idx)
        self.test_ds = Subset(test, range(3000)) if self.quick else test
        tag = "" if self.holdout is None else f" (no {self.holdout})"
        print(f"  classwise train images{tag}: {len(self.train_ds)}")

    def train_dataloader(self):
        # drop_last: a tiny final batch can have every class below MIN_PER_CLASS,
        # which makes the class-conditional SIGReg term a no-grad zero.
        return DataLoader(self.train_ds, shuffle=True, drop_last=True, **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.test_ds, shuffle=False, **self.loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_ds, shuffle=False, **self.loader_kwargs)


class TwoViewMNISTDataModule(GenericDataModule):
    """
    Two augmented views per image.  ``labeled=True`` also returns the label
    (SupCon); ``holdout`` optionally removes a digit from both splits.
    """

    def __init__(self, quick=False, labeled=False, holdout=None, **kwargs):
        super().__init__(**kwargs)
        self.quick = quick
        self.labeled = labeled
        self.holdout = holdout

    def prepare_data(self):
        datasets.MNIST(DATA_DIR, train=True, download=True)
        datasets.MNIST(DATA_DIR, train=False, download=True)

    def _wrap(self, raw, n):
        idx = [i for i in range(n)
               if (self.holdout is None or int(raw.targets[i]) != self.holdout)]
        base = Subset(raw, idx)
        cls = dutils.TwoViewLabeledMNIST if self.labeled else dutils.TwoViewMNIST
        return cls(base)

    def setup(self, stage=None):
        raw_train = datasets.MNIST(DATA_DIR, train=True, transform=None)
        raw_test = datasets.MNIST(DATA_DIR, train=False, transform=None)
        n_train = 8000 if self.quick else len(raw_train)
        n_test = 2000 if self.quick else len(raw_test)
        self.train_ds = self._wrap(raw_train, n_train)
        self.val_ds = self._wrap(raw_test, n_test)
        tag = "" if self.holdout is None else f" (no {self.holdout})"
        print(f"  two-view train images{tag}: {len(self.train_ds)}")

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, drop_last=self.labeled,
                          **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_ds, shuffle=False, drop_last=self.labeled,
                          **self.loader_kwargs)


# --------------------------------------------------------------------------- #
# JetClass DataModules                                                         #
#                                                                             #
# These mirror the MNIST modules above and yield the *same* batch formats, so  #
# the (dataset-agnostic) Lightning modules work unchanged:                     #
#   plain      -> (features, label)                                            #
#   two-view   -> (view1, view2) or (view1, view2, label)                      #
# JetClass ROOT files are read from JETCLASS_DIR (or `data_dir`); a missing     #
# path raises a clear error.                                                    #
# --------------------------------------------------------------------------- #
class _JetClassBase(GenericDataModule):
    """Shared JetClass loading from the real ROOT files."""

    def __init__(self, classes=None, n_particles=jc.N_PARTICLES, quick=False,
                 data_dir=None, **kwargs):
        super().__init__(**kwargs)
        names = classes or jc.DEFAULT_CLASSES        # five classes by default
        self.class_indices = [jc.JETCLASS_CLASSES.index(c) for c in names]
        self.n_classes = len(names)
        self.n_particles = n_particles
        self.quick = quick
        self.data_dir = data_dir or JETCLASS_DIR

    def _load(self, split):
        max_events = (2000 if self.quick else None)
        out = jc.load_root(self.data_dir, split, self.class_indices,
                           n_particles=self.n_particles, max_events=max_events)
        if out is None:
            raise FileNotFoundError(
                f"No JetClass ROOT files for split '{split}' under {self.data_dir!r}. "
                f"Set data_dir (or the JETCLASS_DIR env var) to a directory with "
                f"train/ val/ test/ subfolders of *.root files.")
        print(f"  jetclass[{split}] ROOT: {len(out[1])} jets")
        return out


class JetClassDataModule(_JetClassBase):
    """Plain JetClass ``(features, label)`` for all requested classes."""

    def setup(self, stage=None):
        self.train_X, self.train_y = self._load("train")
        self.val_X, self.val_y = self._load("val")
        self.test_X, self.test_y = self._load("test")

    def train_dataloader(self):
        return DataLoader(jc.JetDataset(self.train_X, self.train_y),
                          shuffle=True, **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(jc.JetDataset(self.val_X, self.val_y),
                          shuffle=False, **self.loader_kwargs)

    def test_dataloader(self):
        return DataLoader(jc.JetDataset(self.test_X, self.test_y),
                          shuffle=False, **self.loader_kwargs)


class JetClassClasswiseDataModule(_JetClassBase):
    """JetClass ``(features, label)``; optionally drops ``holdout`` from training."""

    def __init__(self, holdout=None, **kwargs):
        super().__init__(**kwargs)
        self.holdout = holdout

    def setup(self, stage=None):
        X, y = self._load("train")
        if self.holdout is not None:
            keep = y != self.holdout
            X, y = X[keep], y[keep]
            print(f"  jetclass train without class {self.holdout}: {len(y)} jets")
        self.train_X, self.train_y = X, y
        self.val_X, self.val_y = self._load("val")
        self.test_X, self.test_y = self._load("test")

    def train_dataloader(self):
        # drop_last: see ClasswiseMNISTDataModule -- avoids an all-below-threshold
        # final batch collapsing the class-conditional SIGReg term to a no-grad zero.
        return DataLoader(jc.JetDataset(self.train_X, self.train_y),
                          shuffle=True, drop_last=True, **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(jc.JetDataset(self.val_X, self.val_y),
                          shuffle=False, **self.loader_kwargs)

    def test_dataloader(self):
        return DataLoader(jc.JetDataset(self.test_X, self.test_y),
                          shuffle=False, **self.loader_kwargs)


class JetClassTwoViewDataModule(_JetClassBase):
    """
    Two augmented views per jet.  ``labeled=True`` also returns the label
    (SupCon); ``holdout`` optionally removes a class from both splits.
    """

    def __init__(self, labeled=False, holdout=None, **kwargs):
        super().__init__(**kwargs)
        self.labeled = labeled
        self.holdout = holdout

    def _wrap(self, X, y):
        if self.holdout is not None:
            keep = y != self.holdout
            X, y = X[keep], y[keep]
        return (jc.TwoViewLabeledJets(X, y) if self.labeled else jc.TwoViewJets(X))

    def setup(self, stage=None):
        Xtr, ytr = self._load("train")
        Xval, yval = self._load("val")
        self.train_ds = self._wrap(Xtr, ytr)
        self.val_ds = self._wrap(Xval, yval)

    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, drop_last=self.labeled,
                          **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_ds, shuffle=False, drop_last=self.labeled,
                          **self.loader_kwargs)
