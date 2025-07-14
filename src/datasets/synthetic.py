import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import numpy as np 
import os
import open3d as o3d

class Synthetic(LightningDataModule):

    def __init__(self, opts): # opts is a dictionary 
        super().__init__()
        self.opts = opts
        self.setup()

    def setup(self, stage=None):
        self.data = SyntheticData(self.opts)

    def train_dataloader(self):
        loader = DataLoader(self.data,
                            batch_size = self.opts['batch_size'],
                            num_workers =  self.opts['workers'],
                            pin_memory=True,
                            shuffle=True)
        self.len = self.data_train.__len__()
        return loader

    def val_dataloader(self):
        return self.train_dataloader()

    def test_dataloader(self):
        return self.train_dataloader()

class SyntheticData(Dataset):
    def __init__(self, opts):
        super().__init__()
        self.cloud_path = opts["root"]
        self.clouds = [ x for x in os.listdir(self.cloud_path) if ".ply" in x or ".xyz" in x]
        self.clouds.sort()
        self.len = len(self.clouds)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        sample = {}
        cloud = o3d.io.read_point_cloud(os.path.join(self.cloud_path, self.clouds[index])) 
        sample['mesh'] = torch.tensor(np.array(cloud.points)).type(torch.float) # we call it mesh to be consistent with bbc
        return sample
