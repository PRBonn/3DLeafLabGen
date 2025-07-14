import click
import torch
import yaml

from metrics.FID import FID
from metrics.MMD import MMD
from metrics.improved_pr import IPR
from datasets import get_dataset


@click.command()
@click.option("--config", "-c", type=str, help="path to config file", default="config/config_bbc.yaml")

def main(config):
    cfg = yaml.safe_load(open(config))
    data_name = cfg["data"]["name"]

    # Load data 
    data = get_dataset(data_name , cfg['data']['opts'])
    X = [ torch.tensor(x['mesh']).cuda() for  x in data.data_train ]
   
    filename = "./metrics/" + data_name + ".pt"
    out_dict = {'ipr': {'pointmlp' : {}, 'pointnet' : {}},
                'cmmd': {'pointmlp' : {}, 'pointnet' : {}},
                'fid': {'pointmlp' : {}, 'pointnet' : {}}}
    for model in ["pointmlp", "pointnet"]:
        mmd = MMD(function=model)
        fid = FID(function=model)
        ipr = IPR(function=model, knn= [2, 4, 8, 16, 32, 48, 64, 96]  )
        try:
            y = ipr.compute_target(X)
            out_dict["ipr"][model]["y"] = y
        except:
            print(f'No ipr target computed for model {model}.')
        try:
            y = mmd.compute_target(X)
            out_dict["cmmd"][model]["y"] = y
        except:
            print(f'No cmmd target computed for model {model}.')
        try:
            mean, cov = fid.compute_target(X)    
            out_dict["fid"][model]["mean"] = mean
            out_dict["fid"][model]["cov"] = cov
        except:
            print(f'No fid target computed for model {model}.')

    torch.save(out_dict, filename)

if __name__ == "__main__":
    main()

