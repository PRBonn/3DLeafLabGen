import numpy as np
import torch

class MMD():
    def __init__(self, function="pointmlp", checkpoint="metrics/pointmlp_8k.pth", batch_size = 1):
        from models import get_model
        self.clip = function == "pointmlp"
        self.model = function
        self.forward = get_model(self.model, None).cuda()
        try:
            if self.model == "pointmlp":
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

    def compute_with_target(self, X, target):
        with torch.no_grad():
            x = self.compute_forward_pass(X)
        
        y = target['cmmd'][self.model]['y'].clone().detach().cuda()
        x_sqnorms = torch.diag(torch.matmul(x, x.T))
        y_sqnorms = torch.diag(torch.matmul(y, y.T))

        gamma = 1 / (2 * 10**2)
        k_xx = torch.mean(
          torch.exp(
              -gamma
              * (
                  -2 * torch.matmul(x, x.T)
                  + torch.unsqueeze(x_sqnorms, 1)
                  + torch.unsqueeze(x_sqnorms, 0)
              )
          )
        )
        k_xy = torch.mean(
          torch.exp(
              -gamma
              * (
                  -2 * torch.matmul(x, y.T)
                  + torch.unsqueeze(x_sqnorms, 1)
                  + torch.unsqueeze(y_sqnorms, 0)
              )
          )
        )
        k_yy = torch.mean(
          torch.exp(
              -gamma
              * (
                  -2 * torch.matmul(y, y.T)
                  + torch.unsqueeze(y_sqnorms, 1)
                  + torch.unsqueeze(y_sqnorms, 0)
              )
          )
        )
        return 1000 * (k_xx + k_yy - 2 * k_xy)

    def forward_pointnet(self,X):
        n_batches = int(np.ceil(len(X) / self.batch_size))
        out = []
        for b in range(n_batches):
            if len(X[b].shape) < 3:
                X[b] = X[b].unsqueeze(0)
            if X[b].shape[-1] == 3:
                X[b] = X[b].permute(0,2,1)
            try:
                _, _, item = self.forward(X[b].type(torch.float))
                out.append(item)
            except:
                continue
        return torch.vstack(out)

    def distance(self, x, y):
        x_sqnorms = torch.diag(torch.matmul(x, x.T))
        y_sqnorms = torch.diag(torch.matmul(y, y.T))

        gamma = 1 / (2 * 10**2)
        k_xx = torch.mean(
          torch.exp(
              -gamma
              * (
                  -2 * torch.matmul(x, x.T)
                  + torch.unsqueeze(x_sqnorms, 1)
                  + torch.unsqueeze(x_sqnorms, 0)
              )
          )
        )
        k_xy = torch.mean(
          torch.exp(
              -gamma
              * (
                  -2 * torch.matmul(x, y.T)
                  + torch.unsqueeze(x_sqnorms, 1)
                  + torch.unsqueeze(y_sqnorms, 0)
              )
          )
        )
        k_yy = torch.mean(
          torch.exp(
              -gamma
              * (
                  -2 * torch.matmul(y, y.T)
                  + torch.unsqueeze(y_sqnorms, 1)
                  + torch.unsqueeze(y_sqnorms, 0)
              )
          )
        )
        return 1000 * (k_xx + k_yy - 2 * k_xy)

    def compute_target(self, X):
        with torch.no_grad():
            X_out = self.compute_forward_pass(X)
        return X_out

    def __call__(self, X, Y):
        # compute features
        with torch.no_grad():
            X_out = self.compute_forward_pass(X)
            Y_out = self.compute_forward_pass(Y)
        return self.distance(X_out, Y_out)
