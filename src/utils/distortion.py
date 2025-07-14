import numpy as np
import torch
import random

class Distortion:
    def __init__(self, probability, max_angles, device="gpu"):
        self.p = probability
        self.max_angles = max_angles
        self.device = device

    def __call__(self, points):
        if random.random() > self.p: 
            return points
        
        if self.device == "cpu":
            max_distortion_angles = (np.random.rand(1,3) * self.max_angles)[0]
            points -= np.mean(points, axis=0) 

            # compute distance from mean, which is now 0
            d = ((points**2).sum(axis=1))**0.5
            
            # compute angles for each point
            alphas = max_distortion_angles[0] * d
            betas = max_distortion_angles[1] * d
            gammas = max_distortion_angles[2] * d

            R = np.array([ [ np.cos(betas)*np.cos(gammas) , np.sin(alphas)*np.sin(betas)*np.cos(gammas) - np.cos(alphas)*np.sin(gammas) , np.cos(alphas)*np.sin(betas)*np.cos(gammas) + np.sin(alphas)*np.sin(gammas) ] , [ np.cos(betas)*np.sin(gammas) , np.sin(alphas)*np.sin(betas)*np.sin(gammas) + np.cos(alphas)*np.cos(gammas) , np.cos(alphas)*np.sin(betas)*np.sin(gammas) - np.sin(alphas)*np.cos(gammas)] , [ - np.sin(betas) , np.sin(alphas)*np.cos(betas) , np.cos(alphas)*np.cos(betas)] ])
            R = torch.tensor(R).cuda()
            try:
                points = torch.tensor(points).cuda()
                new_points = (R.T @ points.T)[torch.arange(0, len(points)).cuda(),:, torch.arange(0, len(points)).cuda()]
            except RuntimeError:
                print("point cloud is too big for distortion")
                return points.cpu()
            return np.asarray(new_points.cpu())
        else:
            ## not working bc torch has the weird inability to make multiple tensor one
            max_distortion_angles = (torch.rand((1,3)).cuda() * self.max_angles)[0]
            points -= points.mean(0) 

            # compute distance from mean, which is now 0
            d = ((points**2).sum(1))**0.5
            
            # compute angles for each point
            alphas = max_distortion_angles[0] * d
            betas = max_distortion_angles[1] * d
            gammas = max_distortion_angles[2] * d

            R = torch.tensor([ [ torch.cos(betas)*torch.cos(gammas) , torch.sin(alphas)*torch.sin(betas)*torch.cos(gammas) - torch.cos(alphas)*torch.sin(gammas) , torch.cos(alphas)*torch.sin(betas)*torch.cos(gammas) + torch.sin(alphas)*torch.sin(gammas) ] , [ torch.cos(betas)*torch.sin(gammas) , torch.sin(alphas)*torch.sin(betas)*torch.sin(gammas) + torch.cos(alphas)*torch.cos(gammas) , torch.cos(alphas)*torch.sin(betas)*torch.sin(gammas) - torch.sin(alphas)*torch.cos(gammas)] , [ - torch.sin(betas) , torch.sin(alphas)*torch.cos(betas) , torch.cos(alphas)*torch.cos(betas)] ])
            try:
                new_points = (R.T @ points.T)[torch.arange(0, len(points)).cuda(),:, torch.arange(0, len(points)).cuda()]
                return new_points
            except RuntimeError:
                return points

