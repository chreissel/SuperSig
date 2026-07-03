"""Global configuration and paths for the NLLReg experiments."""
import os
import torch

# Repository layout ---------------------------------------------------------- #
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PKG_DIR)
# MNIST is downloaded here.  Kept out of the `data/` Python package (which now
# holds the Lightning DataModules) so the two never collide.
DATA_DIR = os.path.join(REPO_DIR, "mnist_data")
# Base directory holding the JetClass ROOT files; override with the JETCLASS_DIR
# environment variable.  Expected to contain the train_100M/ val_5M/ test_20M
# subdirectories (the reference cluster layout).
JETCLASS_DIR = os.environ.get(
    "JETCLASS_DIR", "/n/holystore01/LABS/iaifi_lab/Lab/sambt/JetClass/")
PLOTS_DIR = os.path.join(REPO_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Problem / model constants -------------------------------------------------- #
EMB_DIM = 16
N_CLASSES = 10
HOLDOUT = 4                      # digit held out of the embedding in the holdout study

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_path(name):
    """Absolute path for a figure inside the plots/ directory."""
    return os.path.join(PLOTS_DIR, name)
