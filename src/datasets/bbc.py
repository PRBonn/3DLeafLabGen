import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import numpy as np 
import open3d as o3d
import random
import csv
import os 
import functools
from sklearn.mixture import GaussianMixture

class BonnBeetClouds(LightningDataModule):

    def __init__(self, opts): # opts is a dictionary 
        super().__init__()
        self.opts = opts
        self.setup()

    def setup(self, stage=None):
        self.data_train = BBCData(self.opts['train'], 'train')
        self.data_val = BBCData(self.opts['val'], 'val')

    def train_dataloader(self):
        loader = DataLoader(self.data_train,
                            batch_size = self.opts['train']['batch_size'] // self.opts['train']['n_gpus'],
                            num_workers = self.opts['train']['workers'],
                            collate_fn = self.my_collate_function,
                            drop_last = True,
                            shuffle=True)
        self.len = self.data_train.__len__()
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.data_val,
                            batch_size = self.opts['train']['batch_size'] // self.opts['train']['n_gpus'],
                            num_workers = self.opts['train']['workers'],
                            collate_fn = self.my_collate_function,
                            drop_last = True,
                            shuffle=False)
        self.len = self.data_val.__len__()
        return loader

    def my_collate_function(self, items):
        batch = {}
        skeleton = []
        edges = []
        faces = []
        meshes = []
        for i in range(len(items)):
            skeleton.append(torch.tensor(items[i]['skeleton']).type(torch.float).cuda())
            edges.append(torch.tensor(items[i]['edges']).type(torch.int).cuda())
            faces.append(torch.tensor(items[i]['faces']).type(torch.int).cuda())
            meshes.append(torch.tensor(items[i]['mesh']).type(torch.float).cuda())
        batch['skeleton'] = list(skeleton)
        batch['edges'] = list(edges)
        batch['faces'] = list(faces)
        batch['meshes'] = list(meshes)
        return batch

    def test_dataloader(self):
        return self.val_dataloader()

class BBCData(Dataset):
    def __init__(self, opts, split):
        super().__init__()
        self.root = os.path.join(opts["root"], split)
        self.leaves = os.listdir(os.path.join(self.root, "leaves")) 
        self.skeletons = os.listdir(os.path.join(self.root, "skeletons")) 
        self.len = len(self.leaves)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # get mesh
        leaf_path = os.path.join( os.path.join(self.root, "leaves") , self.leaves[index])
        skel_path = os.path.join( os.path.join(self.root, "skeletons") , self.skeletons[index])
        
        leaf = o3d.io.read_point_cloud(leaf_path)
        skel = o3d.io.read_point_cloud(skel_path)
        
        leaf_vert = np.array(leaf.points)
        skeleton_vert = np.array(skel.points)

        # add extra points as fitted plane
        skeleton_vert -= skeleton_vert.mean(0) # center points
        n_total_points = skeleton_vert.shape[0] * 10
        gm = GaussianMixture(n_components=2, random_state=1234, max_iter=50).fit(skeleton_vert)
        extras, _ = gm.sample(n_total_points)
        input_points = np.concatenate((skeleton_vert, extras))
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(input_points)
        
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
        radii = [0.005, 0.01, 0.02, 0.04]
        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        rec_mesh.compute_adjacency_list()       

        edges = []
        done = []
        for point_1 in range(len(rec_mesh.adjacency_list)):
            points = rec_mesh.adjacency_list[point_1]
            for point_2 in points:
                if point_2 not in done:
                    edges.append(np.array([point_1,point_2]))
            done.append(point_1)

        edges = np.vstack(edges)
        
        sample = {}
        sample["mesh"] = leaf_vert
        sample["skeleton"] = input_points
        sample["faces"] = np.asarray(rec_mesh.triangles)
        sample["edges"] = edges
        return sample

