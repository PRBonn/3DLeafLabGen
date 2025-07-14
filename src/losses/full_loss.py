import torch
from pytorch3d.loss import chamfer_distance
from losses.regularization_loss import mesh_edge_loss, mesh_laplacian_smoothing 

class FullLoss:
    def __init__(self, weights=torch.ones(3), skel_size=31):
        self.weights = weights  
        self.skel_size = skel_size

    def __call__(self, offset, in_points, faces, edges, target):
        fxl = torch.tensor([0.0]).cuda()
        cl = torch.tensor([0.0]).cuda()
        pr = torch.tensor([0.0]).cuda()
        final = []
        for batch_item in range(len(offset)):
            fxl += ((offset[batch_item][:self.skel_size])**2).sum() # skeleton needs no offset
            final.append(offset[batch_item] + in_points[batch_item])
            cl += chamfer_distance(final[-1].unsqueeze(0), target[batch_item].unsqueeze(0))[0] * self.weights[0]

        mel = mesh_edge_loss(final, edges) * self.weights[1]
        mls = mesh_laplacian_smoothing(final, edges, faces) * self.weights[2]
        
        return fxl, cl, mel, mls
