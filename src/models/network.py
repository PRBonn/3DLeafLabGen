import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import JaccardIndex as IoU
import os 
import open3d as o3d
import csv
from models.kkpconv.blocks import DecoderBlock

from losses.full_loss import FullLoss
from losses.distribution_loss import DistributionLoss
from utils.utils import save_output
torch.autograd.set_detect_anomaly(True)

class Network(LightningModule):
    def __init__(self, opts):
        super().__init__()
        self.model = opts['model']
        self.optimizer = self.configure_optimizers()
        self.loss = FullLoss(weights=torch.tensor([200,10,1]), skel_size=opts['skel_size'])
        self.dloss = DistributionLoss(knn=opts["knn"], target=opts["data"])
        self.val_error = torch.tensor([0.0]).cuda()       
        self.fxl_error = torch.tensor([0.0]).cuda()       
        self.fid = torch.tensor([0.0]).cuda()       
        self.cmmd = torch.tensor([0.0]).cuda()       
        self.ipr = torch.tensor([0.0]).cuda()       
        self.chamfer_error = torch.tensor([0.0]).cuda()       
        self.mel_error = torch.tensor([0.0]).cuda()       
        self.msl_error = torch.tensor([0.0]).cuda()       
        try:
            folder = len(os.listdir("./experiments/generated"))
        except:
            os.makedirs("./experiments/generated", exist_ok=True)
            folder = '0'
        self.folder_path = os.path.join("./experiments/generated", str(folder))

    def getLoss(self, z:torch.Tensor, skeleton, target, faces, edges):
        fxl, cl, mel, msl = self.loss(z, skeleton, faces, edges, target)
        final = []
        for elem in range(len(z)):
            final.append(z[elem] + skeleton[elem])
        dl = self.dloss(final)
        return fxl, cl, mel, msl, dl["fid"] , dl['cmmd'], dl['ipr']

    def forward(self, x:torch.Tensor):
        grid, out = self.model.forward(x,x)
        return out 
        
    def training_step(self, batch, batch_idx):
        y = self.forward(batch['skeleton'])
        loss = self.getLoss(y, batch['skeleton'], batch['meshes'], batch['faces'], batch['edges'])
        self.log("Loss/total", torch.vstack(loss).sum()/len(batch['skeleton']), True)
        self.log("Loss/fxl", loss[0]/len(batch['skeleton']))
        self.log("Loss/chamfer", loss[1]/len(batch['skeleton']))
        self.log("Loss/edges", loss[2]/len(batch['skeleton']))
        self.log("Loss/laplacian", loss[3]/len(batch['skeleton']))
        self.log("Loss/fid", loss[4] )
        self.log("Loss/cmmd", loss[5] )
        self.log("Loss/ipr", loss[6] )
        return loss[1]*10 + loss[2] + loss[3] + loss[4] * 0.1 + loss[5] * 0.01 + loss[6] * 0.1

    def validation_step(self, batch, batch_idx):
        y = self.forward(batch['skeleton'])
        loss = self.getLoss(y, batch['skeleton'], batch['meshes'], batch['faces'], batch['edges'])
        self.val_error += torch.vstack(loss).sum() / len(batch['skeleton'])
        self.fxl_error += loss[0] / len(batch['skeleton'])
        self.chamfer_error += loss[1] / len(batch['skeleton'])
        self.mel_error += loss[2] / len(batch['skeleton'])
        self.msl_error += loss[3] / len(batch['skeleton'])
        self.fid += loss[4]
        self.cmmd += loss[5]
        self.ipr += loss[6]

    def on_validation_epoch_end(self):
        n_batches = self.trainer.num_val_batches[0]
        self.val_error /= n_batches
        self.fxl_error /= n_batches
        self.chamfer_error /= n_batches
        self.mel_error /= n_batches
        self.msl_error /= n_batches
        self.fid /= n_batches
        self.cmmd /= n_batches
        self.ipr /= n_batches
        self.log("val:loss", self.val_error)        
        self.log("Val/fxl", self.fxl_error)        
        self.log("Val/chamfer", self.chamfer_error)        
        self.log("Val/mel", self.mel_error)        
        self.log("Val/lmsl", self.msl_error)        
        self.log("Val/fid", self.fid)        
        self.log("Val/cmmd", self.cmmd)        
        self.log("Val/ipr", self.ipr)        
        self.val_error *= 0.0 
        self.fxl_error *= 0.0
        self.chamfer_error *= 0.0
        self.mel_error *= 0.0
        self.msl_error *= 0.0 
        self.fid *= 0.0
        self.cmmd *= 0.0
        self.ipr *= 0.0

    def test_step(self, batch, batch_ids):
        y = self.forward(batch['skeleton'])
        final_points = [ torch.vstack(y[x]) + batch['skeleton'][x] for x in range(len(y)) ]
        save_output(self.folder_path, final_points, batch['stem'], batch['len'], batch['wid']) 

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        return [ self.optimizer ]

