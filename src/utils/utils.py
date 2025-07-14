import math
import os
import numpy as np 
import open3d as o3d 
import csv 

def save_output(folder_path, points, stems=[None], lens=[None], wids=[None]):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    id_0 = len(os.listdir(folder_path))
    for _id, cloud in enumerate(points):
        cl = o3d.geometry.PointCloud()
        name = str(_id + id_0) + ".ply"
        cl.points = o3d.utility.Vector3dVector(np.array(cloud[0].cpu()))        
        o3d.io.write_point_cloud(os.path.join(folder_path, str(name)), cl)
        if stems[0] != None: # if generate without need of labels, you can avoid this 
            with open(os.path.join(folder_path,'measures.csv'),"a+") as fil:
                writer = csv.writer(fil)
                writer.writerow([name, stems[_id], lens[_id], wids[_id]])


def rotation_matrix_from_euler(z, y, x):
    mat = np.zeros((3,3))
    ca = np.cos(z)
    sa = np.sin(z)
    cb = np.cos(y)
    sb = np.sin(y)
    cg = np.cos(x)
    sg = np.sin(x)
    
    mat[0,0] = ca * cb
    mat[0,1] = ca*sb*sg - sa*cg
    mat[0,2] = ca*sb*cg + sa*sg
    mat[1,0] = sa*cb
    mat[1,1] = sa*sb*sg + ca*cg 
    mat[1,2] = sa*sb*cg - ca*sg
    mat[2,0] = -sb
    mat[2,1] = cb*sg
    mat[2,2] = cb*cg
    
    return mat
