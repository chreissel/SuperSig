# SuperSig

Learning structured 16-dimensional MNIST embeddings with **SIGReg** (Sketched
Isotropic Gaussian Regularization) and **supervised contrastive learning (SupCon /
supervised SimCLR)**, then evaluating them with frozen linear probes, ROC curves, and
corner plots of the latent space.

The unifying idea is a *distributional* prior on the embedding: SIGReg pushes the
learned features toward a (class-conditional) Gaussian — an isotropic-Gaussian /
negative-log-likelihood style regularizer — rather than relying only on a
discriminative loss. Every embedding is trained, then **frozen**, and a single linear
layer is trained on top with categorical cross-entropy.

The project is organized around **PyTorch Lightning**, following the structure of
[phlab-neurips25](https://github.com/sambt/phlab-neurips25): a training run is fully
described by a YAML config passed to the Lightning CLI (`cli.py`), which wires
together a `LightningModule` (the objective), a `LightningDataModule` (the data +
augmentations), and the trainer / logger / checkpointing settings.

## Method summary

| Embedding | Idea |
|-----------|------|
| Supervised baseline | CNN trained end-to-end with cross-entropy (reference) |
| SIGReg (SSL) | invariance between two augmented views + a global isotropic-Gaussian SIGReg term (no labels) |
| Class-conditional SIGReg | SIGReg applied per class, pulling each digit to `N(mean_c, I)` |
| &nbsp;&nbsp;· fixed anchors | class means fixed at orthogonal anchors |
| &nbsp;&nbsp;· learnable means | means trained, kept apart by a **hinge separation** term |
| &nbsp;&nbsp;· repulsive means | means trained, kept apart by an **inverse-square repulsion** + shrinkage |
| SupCon | supervised contrastive loss on two augmented views |

Two evaluation protocols:
- **Closed-set:** embedding on all digits → 10-way linear probe → one-vs-rest ROC.
- **Hold-out-4:** embedding trained *without* digit 4 → frozen → binary "4 vs rest"
  linear probe. Tests whether an unseen class still lands in its own latent region.

## Layout

```
cli.py                 LightningCLI entrypoint (python cli.py fit --config ...)
submit.sh              minimal SLURM wrapper around cli.py
configs/               one YAML per experiment (model + data + trainer)
  mnist_*.yaml         supervised / sigreg_ssl / sigreg_classwise_* /
                       sigreg_holdout4_* / supcon / supcon_holdout4
  jetclass_*.yaml      the same nine experiments on JetClass
models/                networks, losses, and Lightning modules
  config.py            paths, constants, device
  networks.py          ConvBackbone, SupervisedCNN  (MNIST)
                       ParticleTransformerModel, SupervisedJetNet, MLP  (JetClass)
  losses.py            sigreg, class-conditional sigreg, separation/repulsion, supcon
  litmodels.py         LightningModules (SupervisedModule, SIGRegSSLModule,
                       ClasswiseSIGRegModule, SupConModule) — dataset-agnostic
data/                  Lightning DataModules
  data_utils.py        MNIST transforms + two-view datasets
  jetclass_data.py     JetClass features, ROOT loader (uproot),
                       jet augmentation, two-view datasets
  datasets.py          MNIST + JetClass DataModules (same batch formats)
utils/                 kept out of the Lightning code paths
  plotting.py          ROC and corner-plot helpers
  eval.py              frozen linear/binary probes + collectors
experiments/           downstream analysis scripts (train/load → probe → plot)
  common.py                dataset-aware factories (--dataset mnist|jetclass)
  01_supervised_baseline.py
  02_sigreg_ssl.py
  03_sigreg_classwise.py   --mode fixed|learnmeans|repulse
  04_holdout4.py           --mode learnmeans|repulse|both
  05_supcon.py
plots/                 all generated figures
```

The loss functions in `models/losses.py` are the heart of the study and are used
verbatim by the Lightning modules — the modules never re-implement an objective.

### Two datasets, one test suite

Every experiment runs on **MNIST** (10 digits) or **JetClass**. Following the
reference repo, JetClass embeddings are trained on a **five-class** subset —
`QCD, Tbqq (ttbar), Wqq, Zqq, Hbb` (`data.jetclass_data.DEFAULT_CLASSES`); pass an
explicit `classes:` list in the config to use all ten. The class count is threaded
through the probe, the ROC, and the SIGReg class-anchors, so the loss functions
stay untouched. Only the *encoder* and *DataModule* differ per dataset:

| | MNIST | JetClass |
|--|-------|----------|
| classes | 10 digits | 5 (QCD, Tbqq, Wqq, Zqq, Hbb) |
| input | 28×28 image | particle cloud `[P, 21]` = 17 features + 4 (px,py,pz,E) |
| encoder | `ConvBackbone` | `ParticleTransformerModel` (ParT) |
| projection head | — | `MLP` on the contrastive modules (SSL / SupCon) |
| augmentation | affine (two views) | η–φ rotation (features + vectors) + pt smearing |

The JetClass side mirrors the reference closely: the vendored **ParticleTransformer**
(`models/parT.py`) as the encoder, the full **17 per-particle features** plus the
`pf_vectors` (px,py,pz,E) used for ParT's pairwise features, the reference's manual
per-feature **standardization**, a **projection head** for the contrastive objectives
(loss on the projection, the encoder embedding used for probing), and a distinct
**train / val / test** split.

JetClass data is read from the real ROOT files (via `uproot`). Set the path in the
config — each `configs/jetclass_*.yaml` has a `data.init_args.data_dir` field
pointing at a directory with `train/`, `val/`, `test/` subfolders of `*.root` files:

```yaml
data:
  class_path: data.datasets.JetClassDataModule
  init_args:
    data_dir: /path/to/JetClass    # real ROOT files (train/ val/ test/)
    classes: [QCD, Tbqq, Wqq, Zqq, Hbb]
```

If `data_dir` is left unset it falls back to the `JETCLASS_DIR` environment variable
(default `jetclass_data/`). The path must contain the ROOT files — a missing path
raises a clear error. The `experiments/` scripts take the path the same way —
`--data-dir /path/to/JetClass` (or the env var).

Mirroring the reference, the JetClass configs cap the data seen per epoch with the
Lightning trainer settings `limit_train_batches: 100` and `limit_val_batches: 20`
(the reference uses the same values) — so only a fraction of the full training set is
used. Adjust or remove those `trainer:` keys in the config to change it.

## Usage

```bash
pip install -r requirements.txt
```

### Training an embedding (Lightning CLI)

A run is defined entirely by its config:

```bash
python cli.py fit --config configs/mnist_supcon.yaml
python cli.py fit --config configs/mnist_sigreg_classwise_repulse.yaml

# the same experiments on JetClass
python cli.py fit --config configs/jetclass_supcon.yaml
python cli.py fit --config configs/jetclass_sigreg_classwise_repulse.yaml
```

Checkpoints and logs are written under `runs/<experiment>/`. You can override any
config value on the command line, e.g. `--trainer.max_epochs 20`.

### Downstream evaluation (frozen probe → ROC / corner)

The `experiments/` scripts train an embedding (a short in-script `Trainer.fit`, or
`--ckpt path/to/last.ckpt` to reuse a CLI run), freeze the encoder, fit a linear
probe, and write figures to `plots/`. Add `--quick` for a fast smoke test.

```bash
# from the repo root — add --dataset jetclass to run any of them on JetClass
python experiments/01_supervised_baseline.py
python experiments/02_sigreg_ssl.py --dataset jetclass
python experiments/03_sigreg_classwise.py --mode repulse --dataset jetclass
python experiments/04_holdout4.py --mode both --dataset jetclass
python experiments/05_supcon.py --dataset jetclass

# reuse a checkpoint trained via the CLI
python experiments/05_supcon.py --ckpt runs/mnist_supcon/.../last.ckpt
```

JetClass figures are written alongside the MNIST ones with a `_jetclass` suffix.

## Results (full runs, MNIST test set)

Closed-set, 10-way probe:

| Model | Probe acc | ROC micro-AUC |
|-------|-----------|---------------|
| Supervised CNN (end-to-end) | 0.990 | 0.9999 |
| SIGReg (SSL) | 0.961 | 0.9991 |
| Class SIGReg, fixed anchors | 0.979 | 0.9996 |
| Class SIGReg, learnable means | 0.976 | 0.9995 |
| Class SIGReg, repulsive means | 0.990 | 0.9999 |
| SupCon (supervised SimCLR) | 0.996 | 0.9999 |

Hold-out-4 detection (digit 4 unseen during embedding), 4-vs-rest AUC:

| Embedding | 4-vs-rest AUC |
|-----------|---------------|
| SupCon (supervised SimCLR) | 0.963 |
| Class SIGReg, learnable means | 0.953 |
| Class SIGReg, repulsive means | 0.887 |

Aggressive separation (repulsion) is best for closed-set accuracy but worse at
placing an *unseen* class in its own region — a closed-set vs open-set trade-off.
SupCon leads on both protocols here.
