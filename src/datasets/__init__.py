from datasets.leaf_skeleton import LeafSkeleton
from datasets.bbc import BonnBeetClouds
from datasets.synthetic import Synthetic

def get_dataset(name, opts):
    if name == 'skeletons':
        return LeafSkeleton(opts)
    elif name=='bbc':
        return BonnBeetClouds(opts)
    elif name == "synthetic":
        return Synthetic(opts)
    else:
        raise ModuleNotFoundError
