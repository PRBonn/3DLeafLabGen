import numpy as np
import torch
from sklearn.mixture import GaussianMixture

class IPR():
    def __init__(self, function="pointnet", checkpoint="metrics/pointnet_on_single_view.pth", batch_size = 1, knn = 3, realism= 0.75):
        from models import get_model
        self.clip = function == "pointmlp"
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
        self.knn = knn
        self.realism = realism

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
                #X[b] /= torch.linalg.norm(X[b])
                item = self.forward(X[b].type(torch.float))
                out.append(item)
            except:
                print(f'Invalid element at index {b}.')
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
                #X[b] /= torch.linalg.norm(X[b])
                _, _, item = self.forward(X[b].type(torch.float32))
                out.append(item)
            except:
                print(f'Invalid element at index {b}.')
                continue
        return torch.vstack(out)

    def compute_with_target(self, X, target):
        with torch.no_grad():
            x_out = self.compute_forward_pass(X)

        y = target["ipr"][self.model]["y"].clone().detach().cuda()
        minlen = min(y.shape[0], x_out.shape[0])
    
        x = x_out[:minlen]
        y = y[:minlen]
        return self.pr(x,y,self.knn)

    def pr(self, x, y, k):
        num_samples = min(len(x),len(y))
        x = x[:num_samples]
        y = y[:num_samples]
        dists_xx = torch.cdist(x, x)
        dists_xy = torch.cdist(x, y)
        diff_down = torch.cdist(x,y,p=2)
        dists_yy = torch.cdist(y, y)
       
        if type(k) is list:
            precision = torch.zeros(len(k))
            recall = torch.zeros(len(k))
            density = torch.zeros(len(k))
            coverage = torch.zeros(len(k))
            for id_, knn in enumerate(k):
                kth_values_xx, idx_xx = torch.kthvalue(dists_xx, knn, dim=0)
                kth_values_xy = torch.kthvalue(dists_xy, knn, dim=0)[0]
                kth_values_yy = torch.kthvalue(dists_yy, knn, dim=0)[0]
            
                diff_up = torch.linalg.norm(x - x[idx_xx],dim=1)
                
                realism = torch.max(diff_up/diff_down, 0)[0] > self.realism
                print(f'When using the {knn}-th element as radius, {(realism.sum()/num_samples)*100} % of samples are valid.')
                if realism.sum() == 0:
                    continue   

                precision[id_] = (kth_values_xy[realism] < kth_values_xx.unsqueeze(1)[realism]).any(0).type(torch.float).mean()         
                recall[id_] = (kth_values_xy[realism] < kth_values_yy[realism]).type(torch.float).mean()
                density[id_] = (1. / float(knn)) * (kth_values_xy[realism] < kth_values_xx.unsqueeze(1)[realism]).type(torch.float).sum(0).mean()
                coverage[id_] = (kth_values_xy[realism].min() < kth_values_xx[realism]).type(torch.float).mean()
            print(f'Precision: {precision}. Recall: {recall}.')
            precision = precision.mean()
            recall = recall.mean()
            density = density.mean()
            coverage = coverage.mean()
        else:
            kth_values_xx = torch.kthvalue(dists_xx, k, dim=0)[0]
            kth_values_xy = torch.kthvalue(dists_xy, k, dim=0)[0]
            kth_values_yy = torch.kthvalue(dists_yy, k, dim=0)[0]
             
            precision = (kth_values_xy < kth_values_xx.unsqueeze(1)).any(0).type(torch.float).mean()         
            recall = (kth_values_xy < kth_values_yy).type(torch.float).mean()
            density = (1. / float(k)) * (kth_values_xy < kth_values_xx.unsqueeze(1)).type(torch.float).sum(0).mean()
            coverage = (kth_values_xy.min() < kth_values_xx).type(torch.float).mean()

        return precision, recall, density, coverage 

    def compute_target(self, X):
        with torch.no_grad():
            X_out = self.compute_forward_pass(X)
        return X_out

    def __call__(self, X, Y):
        # compute features
        num_samples = min(len(X),len(Y))
        with torch.no_grad(): 
            X_out = self.compute_forward_pass(X[:num_samples])
            Y_out = self.compute_forward_pass(Y[:num_samples])
        return self.pr(X_out, Y_out, self.knn)
