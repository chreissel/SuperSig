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
# JetClass DataModules (streaming)                                             #
#                                                                             #
# These mirror the MNIST modules above and yield the *same* batch formats, so  #
# the (dataset-agnostic) Lightning modules work unchanged:                     #
#   plain      -> (features, label)                                            #
#   two-view   -> (view1, view2) or (view1, view2, label)                      #
# ROOT files (from JETCLASS_DIR / `data_dir`) are read lazily in chunks via a   #
# streaming IterableDataset, so peak memory is bounded regardless of the        #
# (~100M-jet) split size.  A missing path raises a clear error.                #
# --------------------------------------------------------------------------- #
class _JetClassBase(GenericDataModule):
    """Shared JetClass streaming loader (bounded-memory, from the real ROOT files)."""

    def __init__(self, classes=None, n_particles=jc.N_PARTICLES, quick=False,
                 data_dir=None, max_files_per_class=None, chunk_size=10000,
                 shuffle_buffer=20000, **kwargs):
        super().__init__(**kwargs)
        names = classes or jc.DEFAULT_CLASSES        # five classes by default
        self.class_indices = [jc.JETCLASS_CLASSES.index(c) for c in names]
        self.n_classes = len(names)
        self.n_particles = n_particles
        self.quick = quick
        self.data_dir = data_dir or JETCLASS_DIR
        # `quick` reads a single file per class; otherwise stream everything
        # available (or a user-set cap).  Streaming keeps memory bounded either way.
        self.max_files_per_class = 1 if quick else max_files_per_class
        self.chunk_size = chunk_size
        self.shuffle_buffer = shuffle_buffer

    def _files(self, split):
        fd = jc.class_file_dict(self.data_dir, split, self.class_indices,
                                max_files_per_class=self.max_files_per_class)
        if fd is None:
            raise FileNotFoundError(
                f"No JetClass ROOT files for split '{split}' under {self.data_dir!r}. "
                f"Set data_dir (or the JETCLASS_DIR env var) to the JetClass base "
                f"directory containing train_100M/ val_5M/ test_20M subfolders of "
                f"*.root files.")
        return fd

    def _stream(self, files, mode, shuffle):
        return jc.JetStream(files, n_particles=self.n_particles, mode=mode,
                            shuffle=shuffle, chunk_size=self.chunk_size,
                            shuffle_buffer=self.shuffle_buffer)

    def _loader(self, stream, drop_last=False):
        # IterableDataset -> shuffle is handled inside the stream (DataLoader
        # shuffle must stay False).
        return DataLoader(stream, drop_last=drop_last, **self.loader_kwargs)


class JetClassDataModule(_JetClassBase):
    """Plain JetClass ``(features, label)`` for all requested classes."""

    def setup(self, stage=None):
        self.ftrain = self._files("train")
        self.fval = self._files("val")
        self.ftest = self._files("test")

    def train_dataloader(self):
        return self._loader(self._stream(self.ftrain, "plain", shuffle=True))

    def val_dataloader(self):
        return self._loader(self._stream(self.fval, "plain", shuffle=False))

    def test_dataloader(self):
        return self._loader(self._stream(self.ftest, "plain", shuffle=False))


class JetClassClasswiseDataModule(_JetClassBase):
    """JetClass ``(features, label)``; optionally drops ``holdout`` from training."""

    def __init__(self, holdout=None, **kwargs):
        super().__init__(**kwargs)
        self.holdout = holdout

    def setup(self, stage=None):
        self.ftrain = self._files("train")
        if self.holdout is not None:
            self.ftrain.pop(self.holdout, None)      # drop the held-out class
        self.fval = self._files("val")
        self.ftest = self._files("test")

    def train_dataloader(self):
        # drop_last: see ClasswiseMNISTDataModule -- avoids an all-below-threshold
        # final batch collapsing the class-conditional SIGReg term to a no-grad zero.
        return self._loader(self._stream(self.ftrain, "plain", shuffle=True), drop_last=True)

    def val_dataloader(self):
        return self._loader(self._stream(self.fval, "plain", shuffle=False))

    def test_dataloader(self):
        return self._loader(self._stream(self.ftest, "plain", shuffle=False))


class JetClassTwoViewDataModule(_JetClassBase):
    """
    Two augmented views per jet.  ``labeled=True`` also returns the label
    (SupCon); ``holdout`` optionally removes a class from both splits.
    """

    def __init__(self, labeled=False, holdout=None, **kwargs):
        super().__init__(**kwargs)
        self.labeled = labeled
        self.holdout = holdout
        self.mode = "twoview_labeled" if labeled else "twoview"

    def _files_no_holdout(self, split):
        fd = self._files(split)
        if self.holdout is not None:
            fd.pop(self.holdout, None)
        return fd

    def setup(self, stage=None):
        self.ftrain = self._files_no_holdout("train")
        self.fval = self._files_no_holdout("val")

    def train_dataloader(self):
        return self._loader(self._stream(self.ftrain, self.mode, shuffle=True),
                            drop_last=self.labeled)

    def val_dataloader(self):
        return self._loader(self._stream(self.fval, self.mode, shuffle=False),
                            drop_last=self.labeled)
