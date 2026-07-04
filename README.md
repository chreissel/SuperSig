# SuperSIG

Learning structured embeddings with **SIGReg** (Sketched
Isotropic Gaussian Regularization) and **supervised contrastive learning (SupCon /
supervised SimCLR)**, then evaluating them with frozen linear probes, ROC curves, and
corner plots of the latent space.

The unifying idea is a *distributional* prior on the embedding: SIGReg pushes the
learned features toward a (class-conditional) Gaussian — an isotropic-Gaussian /
negative-log-likelihood style regularizer — rather than relying only on a
discriminative loss. Every embedding is trained, then **frozen**, and a single linear
layer is trained on top with categorical cross-entropy.

The project is organized around **PyTorch Lightning**: a training run is fully
described by a YAML config passed to the Lightning CLI (`cli.py`), which wires
together a `LightningModule` (the objective), a `LightningDataModule` (the data +
augmentations), and the trainer / logger / checkpointing settings.

## Method summary

| Embedding | Idea |
|-----------|------|
| Supervised baseline | Cross-entropy (reference) |
| SIGReg (SSL) | invariance between two augmented views + a global isotropic-Gaussian SIGReg term (no labels) |
| Class-conditional SIGReg | SIGReg applied per class |
| &nbsp;&nbsp;· fixed anchors | class means fixed at orthogonal anchors |
| &nbsp;&nbsp;· learnable means | means trained, kept apart by a **hinge separation** term |
| &nbsp;&nbsp;· repulsive means | means trained, kept apart by an **inverse-square repulsion** + shrinkage |
| SupCon | supervised contrastive loss on two augmented views |

Two evaluation protocols:
- **Closed-set:** embedding on all classes → 10-way linear probe → one-vs-rest ROC.
- **Hold-out:** embedding trained *without* certain class → frozen → binary classification
  linear probe. Tests whether an unseen class still lands in its own latent region.

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

JetClass data is read from the real ROOT files (via `uproot`), **streamed** lazily
in chunks with a bounded shuffle buffer (`JetStream`, an `IterableDataset`) so peak
memory stays flat regardless of split size — the memory-safe analogue of the
reference's weaver `SimpleIterDataset`. Each `configs/jetclass_*.yaml` has a
`data.init_args.data_dir` field pointing at the JetClass base directory, which must
contain the `train_100M/`, `val_5M/`, `test_20M/` subfolders of `*.root` files (the
reference cluster layout):

```yaml
data:
  class_path: data.datasets.JetClassDataModule
  init_args:
    data_dir: /n/holystore01/LABS/iaifi_lab/Lab/sambt/JetClass/
    classes: [QCD, Tbqq, Wqq, Zqq, Hbb]
```

The default (also `JETCLASS_DIR`'s default) is that cluster path. Override it in the
config, or with the `JETCLASS_DIR` environment variable, or per-experiment with
`--data-dir`. A missing path raises a clear error. (A plain `train/ val/ test/`
layout also works if you point `data_dir` at it.)

Mirroring the reference, the JetClass configs cap the data seen per epoch with the
Lightning trainer settings `limit_train_batches: 100` and `limit_val_batches: 20`
— so only a fraction of the full training set is
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

Checkpoints are written under `runs/<experiment>/`. You can override any config
value on the command line, e.g. `--trainer.max_epochs 20`.

On a SLURM cluster submit the same runs with `submit.sh`, which activates the conda/mamba
env and runs `cli.py fit` on one GPU (logs go to `slurm_logs/`):

```bash
sbatch submit.sh configs/jetclass_supcon.yaml
```

Adjust the `#SBATCH` partition and the `mamba activate` env name at the top of
`submit.sh` to match your account.

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
