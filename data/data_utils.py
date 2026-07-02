"""
Dataset primitives and MNIST transforms shared by the DataModules.

Following the reference layout, the augmentation logic lives *here* (in the
loaders) rather than inside the Lightning modules: a two-view loader simply
returns a pair of independently augmented views, so the same LightningModule
works for any augmentation scheme.
"""
import torch
from torchvision import transforms

NORM = transforms.Normalize((0.1307,), (0.3081,))

TF_PLAIN = transforms.Compose([transforms.ToTensor(), NORM])

TF_AUG = transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(), NORM,
])


class TwoViewMNIST(torch.utils.data.Dataset):
    """Two independently augmented views of each image (no label)."""

    def __init__(self, base, aug=TF_AUG):
        self.base, self.aug = base, aug

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return self.aug(img), self.aug(img)


class TwoViewLabeledMNIST(torch.utils.data.Dataset):
    """Two augmented views of each image plus its label (for SupCon)."""

    def __init__(self, base, aug=TF_AUG):
        self.base, self.aug = base, aug

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, y = self.base[idx]
        return self.aug(img), self.aug(img), y
