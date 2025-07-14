import click
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from models import get_model
from datasets import get_dataset

@click.command()
@click.option("--config", "-c", type=str, help="path to the config file (.yaml)", default="./config/config_bbc.yaml")
@click.option(
    "--weights",
    "-w",
    type=str,
    help="path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.",
    default=None,
)
@click.option(
    "--checkpoint", "-ckpt", type=str, help="path to checkpoint file (.ckpt) to resume training.", default=None
)
def main(config, weights, checkpoint):
    cfg = yaml.safe_load(open(config))
    torch.manual_seed(cfg["experiment"]["seed"])

    # Load data and model
    data = get_dataset(cfg["data"]["name"], cfg['data']['opts'])

    if weights is None:
        model = get_model(cfg['model']['name'], cfg['model']['opts'])
    else:
        model = get_model(cfg['model']['name'], cfg).load_from_checkpoint(weights, hparams=cfg)
    
    # Add callbacks:
    checkpoint_saver = ModelCheckpoint(
        dirpath="experiments/" + cfg["experiment"]["id"],
        monitor="val:loss",
        filename="best",
        mode="min",
        verbose=True,
        save_last=True,
    )

    tb_logger = pl_loggers.TensorBoardLogger("experiments/" + cfg["experiment"]["id"], default_hp_metric=False)

    # Setup trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=cfg["train"]["n_gpus"],
        logger=tb_logger,
        max_epochs=cfg["train"]["max_epoch"],
        callbacks=[checkpoint_saver],
    )
    # Train
    trainer.fit(model, data)


if __name__ == "__main__":
    main()

