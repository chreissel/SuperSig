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

from models.config import DATA_DIR, JETCLASS_DIR, JETCLASS_DATA_CONFIG, HOLDOUT  # noqa: F401
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
# JetClass DataModules (vendored weaver SimpleIterDataset + adapter)           #
#                                                                             #
# These wrap the reference's streaming weaver loader (bounded memory, exact     #
# feature/standardization/padding recipe from the YAML data config) and adapt   #
# its per-jet output into the *same* batch formats as the MNIST modules, so the #
# (dataset-agnostic) SIGReg / SupCon Lightning modules work unchanged:          #
#   plain      -> (features, label)                                            #
#   two-view   -> (view1, view2) or (view1, view2, label)                      #
# ROOT files come from JETCLASS_DIR / `data_dir`; a missing path raises.        #
# --------------------------------------------------------------------------- #
class _JetClassBase(GenericDataModule):
    """Shared JetClass loading via the vendored weaver ``SimpleIterDataset``."""

    def __init__(self, classes=None, quick=False, data_dir=None, data_config=None,
                 max_files_per_class=None, fetch_step=0.01, **kwargs):
        super().__init__(**kwargs)
        self.class_names = list(classes) if classes else list(jc.DEFAULT_CLASSES)
        # Label space is fixed by the data config's `labels.value` list (5 classes).
        self.n_classes = len(jc.DEFAULT_CLASSES)
        self.quick = quick
        self.data_dir = data_dir or JETCLASS_DIR
        self.data_config = data_config or JETCLASS_DATA_CONFIG
        self.max_files_per_class = 1 if quick else max_files_per_class
        # quick: load everything at once (tiny files); else the reference's
        # 1%-of-events streaming fetch.
        self.fetch_step = 1.0 if quick else fetch_step

    def _holdout_name(self, holdout):
        return None if holdout is None else jc.DEFAULT_CLASSES[holdout]

    def _file_dict(self, split, drop_class=None):
        fd = jc.build_file_dict(self.data_dir, split, self.class_names,
                                max_files_per_class=self.max_files_per_class)
        if fd is None:
            raise FileNotFoundError(
                f"No JetClass ROOT files for split '{split}' under {self.data_dir!r}. "
                f"Set data_dir (or the JETCLASS_DIR env var) to the JetClass base "
                f"directory containing train_100M/ val_5M/ test_20M subfolders of "
                f"*.root files.")
        if drop_class is not None:
            fd.pop(drop_class, None)
        return fd

    def _loader(self, file_dict, mode, for_training, drop_last=False):
        ds = jc.make_iter_dataset(file_dict, self.data_config,
                                  for_training=for_training, fetch_step=self.fetch_step)
        return DataLoader(jc.JetClassAdapter(ds, mode=mode), drop_last=drop_last,
                          **self.loader_kwargs)


class JetClassDataModule(_JetClassBase):
    """Plain JetClass ``(features, label)`` for all requested classes."""

    def train_dataloader(self):
        return self._loader(self._file_dict("train"), "plain", for_training=True)

    def val_dataloader(self):
        return self._loader(self._file_dict("val"), "plain", for_training=True)

    def test_dataloader(self):
        return self._loader(self._file_dict("test"), "plain", for_training=False)


class JetClassClasswiseDataModule(_JetClassBase):
    """JetClass ``(features, label)``; optionally drops ``holdout`` from training."""

    def __init__(self, holdout=None, **kwargs):
        super().__init__(**kwargs)
        self.holdout = holdout

    def train_dataloader(self):
        # drop_last: a tiny final batch can have every class below MIN_PER_CLASS,
        # collapsing the class-conditional SIGReg term to a no-grad zero.
        fd = self._file_dict("train", drop_class=self._holdout_name(self.holdout))
        return self._loader(fd, "plain", for_training=True, drop_last=True)

    def val_dataloader(self):
        return self._loader(self._file_dict("val"), "plain", for_training=True)

    def test_dataloader(self):
        return self._loader(self._file_dict("test"), "plain", for_training=False)


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

    def train_dataloader(self):
        fd = self._file_dict("train", drop_class=self._holdout_name(self.holdout))
        return self._loader(fd, self.mode, for_training=True, drop_last=self.labeled)

    def val_dataloader(self):
        fd = self._file_dict("val", drop_class=self._holdout_name(self.holdout))
        return self._loader(fd, self.mode, for_training=True, drop_last=self.labeled)
