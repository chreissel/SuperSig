"""
LightningCLI entrypoint.

A training run is fully described by a YAML config in ``configs/`` -- the model
(a ``LightningModule`` from ``models.litmodels``), the data (a
``LightningDataModule`` from ``data.datasets``), and the trainer / logger /
checkpoint settings.  Run one with::

    python cli.py fit --config configs/mnist_supcon.yaml

As in the reference repo, ``before_instantiate_classes`` keeps the logger's
``save_dir`` in sync with the trainer's ``default_root_dir`` so all outputs for
a run land in one place.
"""
from lightning.pytorch.cli import LightningCLI


class CustomCLI(LightningCLI):
    def before_instantiate_classes(self):
        subcommand = getattr(self.config, "subcommand", None)
        cfg = self.config[subcommand] if subcommand else self.config
        outdir = getattr(cfg.trainer, "default_root_dir", None)
        logger = getattr(cfg.trainer, "logger", None)
        if outdir is not None and logger not in (None, True, False):
            init_args = getattr(logger, "init_args", None)
            if init_args is not None and "save_dir" in init_args:
                init_args.save_dir = outdir


def cli_main():
    CustomCLI(save_config_callback=None)


if __name__ == "__main__":
    cli_main()
