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
  mnist_supervised.yaml
  mnist_sigreg_ssl.yaml
  mnist_sigreg_classwise_{fixed,learnmeans,repulse}.yaml
  mnist_sigreg_holdout4_{learnmeans,repulse}.yaml
  mnist_supcon.yaml
  mnist_supcon_holdout4.yaml
models/                networks, losses, and Lightning modules
  config.py            paths, constants, device
  networks.py          ConvBackbone, SupervisedCNN
  losses.py            sigreg, class-conditional sigreg, separation/repulsion, supcon
  litmodels.py         LightningModules (SupervisedModule, SIGRegSSLModule,
                       ClasswiseSIGRegModule, SupConModule)
data/                  Lightning DataModules
  data_utils.py        MNIST transforms + two-view datasets
  datasets.py          MNIST / classwise / two-view DataModules
utils/                 kept out of the Lightning code paths
  plotting.py          ROC and corner-plot helpers
  eval.py              frozen linear/binary probes + collectors
experiments/           downstream analysis scripts (train/load → probe → plot)
  01_supervised_baseline.py
  02_sigreg_ssl.py
  03_sigreg_classwise.py   --mode fixed|learnmeans|repulse
  04_holdout4.py           --mode learnmeans|repulse|both
  05_supcon.py
plots/                 all generated figures
```

The loss functions in `models/losses.py` are the heart of the study and are used
verbatim by the Lightning modules — the modules never re-implement an objective.

## Usage

```bash
pip install -r requirements.txt
```

### Training an embedding (Lightning CLI)

A run is defined entirely by its config:

```bash
python cli.py fit --config configs/mnist_supcon.yaml
python cli.py fit --config configs/mnist_sigreg_classwise_repulse.yaml
python cli.py fit --config configs/mnist_sigreg_holdout4_repulse.yaml
```

Checkpoints and logs are written under `runs/<experiment>/`. You can override any
config value on the command line, e.g. `--trainer.max_epochs 20`.

### Downstream evaluation (frozen probe → ROC / corner)

The `experiments/` scripts train an embedding (a short in-script `Trainer.fit`, or
`--ckpt path/to/last.ckpt` to reuse a CLI run), freeze the encoder, fit a linear
probe, and write figures to `plots/`. Add `--quick` for a fast smoke test.

```bash
# from the repo root
python experiments/01_supervised_baseline.py
python experiments/02_sigreg_ssl.py
python experiments/03_sigreg_classwise.py --mode repulse
python experiments/04_holdout4.py --mode both
python experiments/05_supcon.py

# reuse a checkpoint trained via the CLI
python experiments/05_supcon.py --ckpt runs/mnist_supcon/.../last.ckpt
```

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
