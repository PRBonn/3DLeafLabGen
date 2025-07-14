import numpy as np
import torch
from sklearn.mixture import GaussianMixture

class FID():
    def __init__(self, function="pointnet", checkpoint="metrics/pointnet_on_single_view.pth", batch_size = 1):
        self.clip = function == "pointmlp"
        from models import get_model
        self.model = function
        self.forward = get_model(function, None).cuda()
        try:
            if function == "pointmlp":
                self.forward.load_model_from_ckpt(self.forward, "metrics/pointmlp_8k.pth")
            else:
                self.forward.load_state_dict(torch.load("metrics/pointnet_on_single_view.pth")["model_state_dict"])
        except:
            raise RuntimeError("Not able to initialize model, check your chekpoint!")
        self.forward.eval()
        self.batch_size = batch_size

    def compute_forward_pass(self, X):
        if self.clip: return self.forward_pointmlp(X)
        return self.forward_pointnet(X)

    def forward_pointmlp(self,X):
        n_batches = int(np.ceil(len(X) / self.batch_size))
        out = []
        for b in range(n_batches):
            if len(X[b].shape) < 3:
                X[b] = X[b].unsqueeze(0)
            if X[b].shape[-1] == 3:
                X[b] = X[b].permute(0,2,1)
            try:
                item = self.forward(X[b].type(torch.float))
                out.append(item)
            except:
                continue
        return torch.vstack(out)

    def forward_pointnet(self,X):
        n_batches = int(np.ceil(len(X) / self.batch_size))
        out = []
        for b in range(n_batches):
            if len(X[b].shape) < 3:
                X[b] = X[b].unsqueeze(0)
            if X[b].shape[-1] == 3:
                X[b] = X[b].permute(0,2,1)
            try:
                X[b] /= torch.linalg.norm(X[b])
                _, _, item = self.forward(X[b].type(torch.float))
                out.append(item)
            except:
                continue
        return torch.vstack(out)

    def distance(self, gmx, gmy):
        mean_x = gmx.means_
        mean_y = gmy.means_

        cov_x = gmx.covariances_**2
        cov_y = gmy.covariances_**2

        dist = (mean_x - mean_y).sum()**2 + np.trace(cov_x + cov_y) - 2*np.sqrt(np.linalg.eigvals(np.matmul(cov_x,cov_y))).sum()

        return dist

    def compute_with_target(self, X, target):
        with torch.no_grad():
            X_out = self.compute_forward_pass(X) 
        x_mean = (X_out).mean(0)
        x_cov = X_out.T.cov()
        target_mean = torch.tensor(target["fid"][self.model]["mean"]).cuda()
        target_cov = torch.tensor(target["fid"][self.model]["cov"]).type(torch.float32).cuda()
        return (x_mean - target_mean).sum()**2 + torch.trace(x_cov + target_cov) - 2*torch.sqrt(torch.linalg.eigvals(torch.mm(x_cov,target_cov))).real.sum()

    def compute_target(self, X):
        with torch.no_grad(): 
            X_out = self.compute_forward_pass(X)
        try:
            gmx = GaussianMixture(covariance_type="tied").fit(np.asarray(X_out.cpu()))
        except:
            # this happens with samples are too similar, no gaussian can be fitted
            print(f"No gaussian was fitted because samples are too similar, no FID target computed.")
            return 0, 0
        return gmx.means_, gmx.covariances_    

    def __call__(self, X, Y):
        # compute features
        with torch.no_grad(): 
            X_out = self.compute_forward_pass(X)
            Y_out = self.compute_forward_pass(Y)
            # fit gaussians
            try:
                gmx = GaussianMixture(covariance_type="tied").fit(np.asarray(X_out.cpu()))
            except:
                # this happens with samples are too similar, no gaussian can be fitted
                print(f"No gaussian was fitted on first set of data. Samples are too similar, no FID target computed.")
                return None
            try:
                gmy = GaussianMixture(covariance_type="tied").fit(np.asarray(Y_out.cpu()))      
            except:
                # this happens with samples are too similar, no gaussian can be fitted
                print(f"No gaussian was fitted on second set of data. Samples are too similar, no FID target computed.")
                return None
        # gaussian distance
        return self.distance(gmx, gmy)
