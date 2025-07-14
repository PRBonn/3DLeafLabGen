import click
import torch
import yaml
import numpy as np

from metrics.FID import FID
from metrics.MMD import MMD
from metrics.improved_pr import IPR
from datasets import get_dataset
import random

@click.command()
@click.option("--config", "-c", type=str, help="path to config file", default="config/test_config.yaml")
@click.option("--model", "-m", type=str, help="name of model to use.", default="pointmlp")
@click.option("--realism", "-r", type=float, help="realism filter value", default=0.5)
@click.option("--num_samples", "-n", type=int, help="num samples to use in the eval", default=1000)

def main(config, model, realism, num_samples):
    cfg = yaml.safe_load(open(config))
    
    # Load data 
    data_source = get_dataset(cfg['data']['source']['name'], cfg['data']['source']['opts'])
    data_target = get_dataset(cfg['data']['target']['name'], cfg['data']['target']['opts'])
   
    # we center all data
    # some models are highly sensitive to 3d translations
    X = [ torch.tensor(x['mesh'] - x["mesh"].mean(0)).cuda() for  x in data_source.data_val ]
    Y = [ (y['mesh'].cuda() - y["mesh"].mean(0).cuda()) for y in data_target.data ]

    max_knn = min(len(X), len(Y))
    # create a list of feasible knns
    knn = [ 2**j for j in range(1, int(np.round(np.log2(max_knn))) ) ]
    
    mmd = MMD(function=model)
    fid = FID(function=model)
    ipr = IPR(function=model, knn=knn , realism=realism )

    prres = ipr(X,Y[:num_samples])
    print(f'IPR: {prres}.\n')    
    dist_mmd = mmd(X,Y[:num_samples])
    print(f'CMMD: {dist_mmd}.\n')     
    dist = fid(X,Y)    
    print(f'FID: {dist}.\n')     


if __name__ == "__main__":
    main()

