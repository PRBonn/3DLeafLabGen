import click
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from models import get_model
from datasets import get_dataset

@click.command()
@click.option("--config", "-c", type=str, help="path to the config file (.yaml)", default="./config/generate_config.yaml")
@click.option(
    "--weights",
    "-w",
    type=str,
    help="path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.",
    default=None,
)
def main(config, weights):
    cfg = yaml.safe_load(open(config))
    torch.manual_seed(cfg["experiment"]["seed"])

    # Load data and model
    data = get_dataset(cfg["data"]["name"], cfg["data"])
    
    if weights is None:
        raise RuntimeError("Cannot test without model weights!")   
    else:
        model = get_model(cfg['model']['name'], cfg['model']['opts'])
        model.load_state_dict(torch.load(weights)['state_dict'])
        model.eval()        
 
    tb_logger = pl_loggers.TensorBoardLogger("experiments/" + cfg["experiment"]["id"], default_hp_metric=False)

    # Setup trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=cfg["train"]["n_gpus"],
        logger=tb_logger,
        max_epochs=cfg["train"]["max_epoch"],
    )
    # Train
    trainer.test(model, data)

if __name__ == "__main__":
    main()

