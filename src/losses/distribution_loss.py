from metrics.FID import FID
from metrics.MMD import MMD
from metrics.improved_pr import IPR
import os
import torch

class DistributionLoss():
    def __init__(self, losses = ['fid','cmmd','ipr'], models = ['pointnet','pointmlp','pointnet'], knn = [2, 4, 8, 16, 32, 64], target = 'bbc'):
        self.models = models
        self.losses = losses
        self.knn = knn
        self.target_distribution = self.get_target_distribution(target)
        self.ipr = IPR(function=self.models[self.losses.index('ipr')], knn=self.knn)
        self.fid = FID(function=self.models[self.losses.index('fid')])
        self.mmd = MMD(function=self.models[self.losses.index('cmmd')])
        self.fid_id = self.losses.index('fid')
        self.mmd_id = self.losses.index('cmmd')
        self.ipr_id = self.losses.index('ipr')

    def get_target_distribution(self, target):
        filename = './metrics/'+target+'.pt'
        if os.path.exists(filename):
            return torch.load(filename)
        else:
            raise ValueError("No target distribution found.\nEither disable the distribution loss or use 'make compute_target' to save the target distribution.")

    def __call__(self, x):
        loss_values = {}
        for loss in self.losses:
            if loss == 'fid':
                loss_values[loss] = self.FIDLoss(x)
            elif loss == 'cmmd':
                loss_values[loss] = self.CMMDLoss(x)
            elif loss == 'ipr':
                loss_dict = self.IPRLoss(x)
                loss_values[loss] = torch.log10(1/(loss_dict[0]+1e-4)) + torch.log10(1/(loss_dict[1]+1e-4))
            else:
                raise ValueError(f'Unknown distribution loss: {loss}.')
        return loss_values

    def IPRLoss(self, x):
        if "y" in self.target_distribution["ipr"][self.models[self.ipr_id]].keys():
            return self.ipr.compute_with_target(x, self.target_distribution) 
        return torch.tensor([0.0]).cuda()

    def CMMDLoss(self, x):
        if "y" in self.target_distribution["cmmd"][self.models[ self.mmd_id ]].keys():
            return self.mmd.compute_with_target(x, self.target_distribution) 
        return torch.tensor([0.0]).cuda()

    def FIDLoss(self, x):
        if "mean" in self.target_distribution["fid"][self.models[ self.fid_id ]].keys() and "cov" in self.target_distribution["fid"][self.models[ self.fid_id ]].keys():
            return self.fid.compute_with_target(x, self.target_distribution) 
        return torch.tensor([0.0]).cuda()
