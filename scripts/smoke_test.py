"""
Super-quick smoke test: exercises the whole stack on synthetic data, so it needs
no MNIST download, no real JetClass files, and no W&B login.

Checks, in order:
  1. imports + all YAML configs parse through the Lightning CLI;
  2. every MNIST Lightning module runs a 1-epoch Trainer.fit on synthetic images;
  3. every JetClass module runs a 1-epoch Trainer.fit through the vendored weaver
     loader + adapter on a handful of synthesized JetClass-style ROOT files.

Run it with `bash scripts/smoke_test.sh` (or `python scripts/smoke_test.py`).
"""
import os
import sys
import glob
import subprocess
import tempfile
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import lightning as pl
from torch.utils.data import DataLoader, TensorDataset, Dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
failures = []


def section(name):
    print(f"\n=== {name} ===")


def trainer():
    return pl.Trainer(max_epochs=1, accelerator="cpu", logger=False,
                      enable_checkpointing=False, num_sanity_val_steps=0,
                      enable_model_summary=False, enable_progress_bar=False,
                      limit_train_batches=2, limit_val_batches=1)


# --------------------------------------------------------------------------- #
# 1. config parsing (two representative configs; loop all if you want)         #
# --------------------------------------------------------------------------- #
def check_configs():
    section("configs parse (LightningCLI --print_config)")
    cfgs = ["configs/mnist_supcon.yaml", "configs/jetclass_sigreg_classwise_repulse.yaml"]
    for c in cfgs:
        r = subprocess.run([sys.executable, "cli.py", "fit", "--config", c, "--print_config"],
                           cwd=REPO, capture_output=True, text=True)
        ok = r.returncode == 0 and "class_path" in r.stdout
        print(f"  {'OK  ' if ok else 'FAIL'} {c}")
        if not ok:
            failures.append(f"config parse {c}\n{r.stderr[-400:]}")


# --------------------------------------------------------------------------- #
# 2. MNIST modules on synthetic images                                         #
# --------------------------------------------------------------------------- #
def check_mnist():
    section("MNIST modules (synthetic images)")
    from models.networks import ConvBackbone, SupervisedCNN
    from models.litmodels import (SupervisedModule, SIGRegSSLModule,
                                  ClasswiseSIGRegModule, SupConModule)
    n = 160
    x = torch.randn(n, 1, 28, 28)
    y = torch.arange(10).repeat(n // 10)               # balanced (>= MIN_PER_CLASS)
    v = torch.randn(n, 1, 28, 28)

    def loader(ds, **kw):
        return DataLoader(ds, batch_size=80, **kw)

    plain = loader(TensorDataset(x, y))
    two = loader(TensorDataset(x, v))
    twolab = loader(TensorDataset(x, v, y), drop_last=True)

    cases = [
        ("supervised", SupervisedModule(SupervisedCNN()), plain),
        ("sigreg-ssl", SIGRegSSLModule(ConvBackbone()), two),
        ("classwise-fixed", ClasswiseSIGRegModule(ConvBackbone(), mode="fixed"), plain),
        ("classwise-repulse", ClasswiseSIGRegModule(ConvBackbone(), mode="repulse"), plain),
        ("supcon", SupConModule(ConvBackbone()), twolab),
    ]
    for name, module, dl in cases:
        try:
            trainer().fit(module, train_dataloaders=dl, val_dataloaders=dl)
            print(f"  OK   {name}")
        except Exception as e:                              # noqa: BLE001
            print(f"  FAIL {name}: {e}")
            failures.append(f"mnist {name}: {e}")


# --------------------------------------------------------------------------- #
# 3. JetClass modules through vendored weaver + adapter (synthetic ROOT)        #
# --------------------------------------------------------------------------- #
def _synth_jetclass(base):
    import awkward as ak
    import uproot
    import data.jetclass_data as jc
    PART = ["part_px", "part_py", "part_pz", "part_energy", "part_deta", "part_dphi",
            "part_d0val", "part_d0err", "part_dzval", "part_dzerr", "part_charge",
            "part_isChargedHadron", "part_isNeutralHadron", "part_isPhoton",
            "part_isElectron", "part_isMuon"]
    OBS = ["jet_eta", "jet_phi", "jet_nparticles", "jet_sdmass",
           "jet_tau1", "jet_tau2", "jet_tau3", "jet_tau4"]
    LABELS = ["label_QCD", "label_Tbqq", "label_Wqq", "label_Zqq", "label_Hbb"]
    rng = np.random.default_rng(0)
    bt = {b: "var * float64" for b in PART}
    bt.update({"jet_pt": np.float64, "jet_energy": np.float64})
    bt.update({o: np.float64 for o in OBS})
    bt.update({L: np.int32 for L in LABELS})

    def make(cls, nj):
        counts = rng.integers(15, 40, size=nj)

        def jag(s=1.0):
            return ak.Array([rng.normal(0, s, int(c)).tolist() for c in counts])

        px, py, pz = jag(20), jag(20), jag(40)
        # physical (timelike) energy E >= |p| so ParT's pairwise log(m^2) is finite
        en = ak.Array([np.sqrt(np.asarray(a) ** 2 + np.asarray(b) ** 2 + np.asarray(c) ** 2) + 0.1
                       for a, b, c in zip(px.tolist(), py.tolist(), pz.tolist())])
        d = {"part_px": px, "part_py": py, "part_pz": pz, "part_energy": en,
             "part_deta": jag(0.3), "part_dphi": jag(0.3), "part_d0val": jag(0.1),
             "part_d0err": jag(0.1), "part_dzval": jag(0.1), "part_dzerr": jag(0.1),
             "part_charge": jag(1), "part_isChargedHadron": jag(1),
             "part_isNeutralHadron": jag(1), "part_isPhoton": jag(1),
             "part_isElectron": jag(1), "part_isMuon": jag(1)}
        d["jet_pt"] = np.hypot(ak.to_numpy(ak.sum(px, 1)), ak.to_numpy(ak.sum(py, 1))) + 1.0
        d["jet_energy"] = ak.to_numpy(ak.sum(en, 1)) + 1.0
        for o in OBS:
            d[o] = rng.normal(0, 1, nj)
        for L in LABELS:
            d[L] = np.full(nj, 1 if L == f"label_{cls}" else 0, dtype=np.int32)
        return d

    for split, nj in [("train_100M", 120), ("val_5M", 60), ("test_20M", 60)]:
        dd = os.path.join(base, split)
        os.makedirs(dd)
        for c in jc.DEFAULT_CLASSES:
            with uproot.recreate(os.path.join(dd, f"{jc.CLASS_FILE_HEADERS[c]}_000.root")) as f:
                f.mktree("tree", bt)
                f["tree"].extend(make(c, nj))


def check_jetclass():
    section("JetClass modules (vendored weaver + synthetic ROOT)")
    try:
        import uproot  # noqa: F401
        import awkward  # noqa: F401
    except ImportError:
        print("  SKIP (uproot/awkward not installed; `pip install uproot awkward`)")
        return
    from models.networks import ParticleTransformerModel, SupervisedJetNet, MLP
    from models.litmodels import (SupervisedModule, SIGRegSSLModule,
                                  ClasswiseSIGRegModule, SupConModule)
    from data.datasets import (JetClassDataModule, JetClassClasswiseDataModule,
                               JetClassTwoViewDataModule)

    def part():   # tiny ParT for speed
        return ParticleTransformerModel(input_dim=17, emb_dim=16, embed_dims=[16, 32, 16],
                                        pair_embed_dims=[8, 8, 8], num_heads=2, num_layers=1,
                                        num_cls_layers=1, fc_params=[[16, 0.0]])

    with tempfile.TemporaryDirectory() as base:
        _synth_jetclass(base)
        kw = dict(data_dir=base, quick=True, batch_size=64, num_workers=0)
        cases = [
            ("supervised", SupervisedModule(SupervisedJetNet(input_dim=17, n_classes=5)),
             JetClassDataModule(**kw)),
            ("sigreg-ssl", SIGRegSSLModule(part(), projector=MLP(16, [16], 16)),
             JetClassTwoViewDataModule(labeled=False, **kw)),
            ("classwise-repulse", ClasswiseSIGRegModule(part(), mode="repulse", n_classes=5),
             JetClassClasswiseDataModule(**kw)),
            ("supcon", SupConModule(part(), projector=MLP(16, [16], 16)),
             JetClassTwoViewDataModule(labeled=True, **kw)),
        ]
        for name, module, dm in cases:
            try:
                trainer().fit(module, dm)
                print(f"  OK   {name}")
            except Exception as e:                          # noqa: BLE001
                print(f"  FAIL {name}: {e}")
                failures.append(f"jetclass {name}: {e}")


def main():
    check_configs()
    check_mnist()
    check_jetclass()
    print("\n" + "=" * 40)
    if failures:
        print(f"SMOKE TEST FAILED ({len(failures)} issue(s)):")
        for f in failures:
            print("  -", f.splitlines()[0])
        sys.exit(1)
    print("SMOKE TEST PASSED ✓")


if __name__ == "__main__":
    main()
