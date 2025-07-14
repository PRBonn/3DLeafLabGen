import models.classification.PointNet2 as PointNet
from models.classification.Clip3D import pointMLPProject
from models.kkpconv.blocks import KPUnet
from models.network import Network

def get_model(name, opts):
    if name == 'kpunet':
        return Network({'model':
                    KPUnet(out_dim = opts['out_dim'],
                    start_voxel_size = opts['start_voxel_size'],
                    init_voxel_size = opts['init_voxel_size'],
                    end_voxel_size = opts['end_voxel_size'] ,
                    in_dim = opts['in_dim'],
                    hidden_feature_dim = opts['hidden_feature_dim'],
                    num_subsamplings = opts['num_subsamplings'],
                    num_stem_layers = opts['num_stem_layers'],
                    conv_radius_scale = opts['conv_radius_scale'],
                    do_deconvolution = opts['do_deconvolution'],
                    random_sample = opts['random_sample'],
                    do_interpolation = True, #opts['do_interpolation'],
                    log_voxel_scaling = opts['log_voxel_scaling'],
                    aggregation_method = opts['aggregation_method']),
                    'skel_size': opts["skel_size"], 
                    'data': opts["data"],
                    'knn': opts["knn"]
                    } )
    elif name == 'pointnet':
        return PointNet.get_model(40)
    elif name == "pointmlp":
        return pointMLPProject()
    else:
        raise ModuleNotFoundError
