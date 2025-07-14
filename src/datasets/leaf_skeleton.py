import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import numpy as np 
import open3d as o3d
import random
from sklearn.mixture import GaussianMixture

from modules.plant import Plant
from utils.utils import rotation_matrix_from_euler as rotation_matrix

class LeafSkeleton(LightningDataModule):

    def __init__(self, opts): 
        super().__init__()
        self.opts = opts
        self.setup()

    def setup(self, stage=None):
        self.data = LeafSkeletonData(self.opts)

    def train_dataloader(self):
        loader = DataLoader(self.data,
                            batch_size = 1, 
                            num_workers = 0,
                            pin_memory=True,
                            shuffle=True)
        self.len = self.data.__len__()
        return loader

    def my_collate(self, items):
        batch = {}
        skeleton = []
        widths = []
        lengths = []
        stems = []
        for i in range(len(items)):
            for j in range(len(items[i]['skeleton'])):
                skeleton.append(torch.tensor(items[i]['skeleton'][j], dtype = torch.float32).cuda())
                widths.append(items[i]['wid'][j])
                lengths.append(items[i]['len'][j])
                stems.append(items[i]['stem'][j])
        batch['skeleton'] = list(skeleton)
        batch['wid'] = list(widths)
        batch['len'] = list(lengths)
        batch['stem'] = list(stems)
        return batch

    def test_dataloader(self):
        return self.train_dataloader()

class LeafSkeletonData(Dataset):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.len = self.opts['data_len']
        self.plants = [ Plant() for _ in range(self.len) ] 
        self.stages = [ p.stage for p in self.plants ]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        sample = {}
        sample['len'] = []
        sample['wid'] = []
        sample['stem'] = []
        current_plant = self.plants[index]
        current_plant.leaves_skeletons = {}
        for id_, l in current_plant.leaves.items():
            skeleton, stem, lenght, width = self.generate_skeleton(l)
            l.length = lenght
            l.width = width
            sample['len'].append(lenght)
            sample['wid'].append(width)
            sample['stem'].append(stem)
            if self.opts["transforms"]:
                skeleton = self.rotate(skeleton)
            current_plant.leaves_skeletons[id_] = skeleton
        
        sample['skeleton'] = list(current_plant.leaves_skeletons.values())
        return sample

    def rotate(self, skeleton):
        # skeleton is a N x 3 array of points
        z_angle = random.random() * np.pi * 2
        y_angle = random.random() * random.choice((-1,1)) * 0.75
        x_angle = random.random() * random.choice((-1,1)) * 0.5
        rotmat = rotation_matrix(z_angle, y_angle, x_angle)
        return (rotmat @ skeleton.T).T

    def generate_skeleton(self, leaf):
        n_points_length = 10  
        n_points_width  = 10 
        
        # keep y fixed, we have x as independet and z as dependetn
        # we first use an hyperbolic tangent 
      
        x_pos = np.linspace(0., 2., n_points_length) * leaf.length 
        y_pos = np.zeros(n_points_length)
        z_pos = np.tanh(x_pos)
        length = np.log(np.cosh(2.)) * leaf.length
                
        # generate second axis
        mid_point_x =  np.random.uniform(0.25, 1.75) * leaf.length
        mid_point = [ mid_point_x, 0., np.tanh(mid_point_x) ] 
        p1 = [ mid_point[0], - 2*leaf.width , np.random.rand() * 0.5 + np.max(z_pos) ]
        p2 = [ mid_point[0], 2* leaf.width  , np.random.rand() * 0.5 + np.max(z_pos) ]
        p3 = mid_point

        a2 = ( p3[-1] - p1[-1] - p2[-1] * p3[1] + p1[-1] * p3[1] - p2[-1] * p1[1] - p1[-1] * p1[1] ) / ( p3[1]**2 * p2[1] - p3[1]**2 * p1[1] + p1[1]**2 * p3[1] - p1[1]**3 - p2[1]**2 * p3[1] + p2[1]**2 * p1[1] - p1[1]**2 )
        b2 = ( p2[-1] - p1[-1] + a2 * p1[1]**2 - a2 * p2[1]**2 ) / ( p2[1] - p1[1] )
        c2 =  p3[-1] - a2 * p3[1]**2 - b2 * p3[1]

        width = abs(a2/3 * (p2[1]**3 - p1[1]**3) + b2/2 * (p2[1]**2 - p1[1]**2) + c2 * (p2[1] -p1[0]))
        # lateral axis points
        x2_pos = np.ones(n_points_width) * mid_point[0]
        y2_pos = np.linspace(0, 1, n_points_width) * 2 * leaf.width  - leaf.width 
        z2_pos = a2 * y2_pos**2 + b2 * y2_pos + c2 


        # stem points
        if self.opts["stem"]:
            leaf_angle = np.random.rand() * np.pi / 4 + np.pi/12
            x3_pos = np.linspace(np.random.uniform(low=-length/4, high=-length/6), 0., n_points_length)  
            y3_pos = np.zeros(n_points_length)
            z3_pos = x3_pos / ( np.tan(leaf_angle) + 1e-2)
            stem_length = np.sqrt(1 + np.tan(leaf_angle)**2) * x3_pos[0]
            length += stem_length

            points = np.concatenate( (np.concatenate((x_pos, x2_pos, x3_pos))[:,None], np.concatenate((y_pos,y2_pos,y3_pos))[:,None], np.concatenate((z_pos,z2_pos,z3_pos))[:,None] ), 1 )
        else:
            stem_length = 0.0
            points = np.concatenate( (np.concatenate((x_pos, x2_pos))[:,None], np.concatenate((y_pos,y2_pos))[:,None], np.concatenate((z_pos,z2_pos))[:,None] ), 1 )
        
        points -= points.mean(0) # center points
        n_total_points = points.shape[0] * 10
        if self.opts["stem"]:
            gm = GaussianMixture(n_components=2, max_iter=25).fit(points)
        else:
            gm = GaussianMixture(n_components=1, max_iter=25).fit(points)
        extras, _ = gm.sample(n_total_points)
        input_points = np.concatenate((points, extras))
        
        return input_points.astype(np.float32), stem_length, length, width

