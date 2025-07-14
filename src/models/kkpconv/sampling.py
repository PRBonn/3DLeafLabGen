import torch
import math
import torch.nn as nn


def meanGridSampling(
    pcd: torch.Tensor,
    resolution_meter,
    scale=1.0,
    features=None,
    smpl_feats=None,
    mask=None,
    fill_value=0,
):
    """Computes the mean over all points in the grid cells

    Args:
        pcd (torch.Tensor): [...,N,3] point coordinates
        resolution_meter ([type]): grid resolution in meters
        scale (float, optional): Defaults to 1.0. Scale to convert resolution_meter to grid_resolution: resolution = resolution_meter/scale
        features (torch.Tensor): [...,N,D] point features
        smpl_feats (torch.Tensor): [...,N,D] additional point features to be grid sampled
    Returns:
        grid_coords (torch.Tensor): [...,N,3] grid coordinates
        grid_features (torch.Tensor): [...,N,D] grid features
        grid_smpl_feats (torch.Tensor): [...,N,D] additional point features to have been grid sampled
        mask (torch.Tensor): [...,N,1] valid point mask (True: valid, False: not valid)
    """
    resolution = resolution_meter / scale
    if len(pcd.shape) < 3:
        pcd = pcd.unsqueeze(0)
    if len(features.shape) < 3:
        features = features.unsqueeze(0)
    B = pcd.shape[0]

    grid_coords = torch.zeros_like(pcd, device=pcd.device)
    grid_features = torch.zeros_like(features, device=pcd.device)
    if smpl_feats != None:
        if len(smpl_feats.shape) < 3:
            smpl_feats = smpl_feats.unsqueeze(0)
        grid_smpl_feats = torch.zeros_like(smpl_feats, device=pcd.device)
    out_mask = torch.full_like(
        pcd[..., :1], False, dtype=bool, device=pcd.device)

    if mask is not None:
        pcd[~mask.expand_as(pcd)] = float("inf")
    grid = torch.floor(
        (pcd - pcd.min(dim=-2, keepdim=True)[0]) / resolution).double()

    if mask is not None:
        pcd[~mask.expand_as(pcd)] = fill_value

    # v_size = math.ceil(1 / resolution)
    # grid_idx = grid[..., 0] + grid[..., 1] * v_size + grid[..., 2] * v_size * v_size
    if mask is not None:
        grid_size = grid[mask.squeeze(-1)].max().detach() + 1
    else:
        grid_size = grid.max().detach() + 1
    grid_idx = (
        grid[..., 0] + grid[..., 1] * grid_size +
        grid[..., 2] * grid_size * grid_size
    )

    max_nr = []
    for i in range(B):
        unique, indices, counts = torch.unique(
            grid_idx[i], return_inverse=True, dim=None, return_counts=True
        )
        indices.unsqueeze_(-1)

        nr_cells = len(counts)
        if unique[-1].isinf():
            counts = counts[:-1]
            nr_cells -= 1
        max_nr.append(nr_cells)

        grid_coords[i].scatter_add_(-2, indices.expand(pcd[i].shape), pcd[i])
        grid_coords[i, :nr_cells, :] /= counts.unsqueeze(-1)

        grid_features[i].scatter_add_(
            -2, indices.expand(features[i].shape), features[i]
        )
        grid_features[i, :nr_cells, :] /= counts.unsqueeze(-1)
        if smpl_feats != None:
            grid_smpl_feats[i].scatter_add_(
                -2, indices.expand(smpl_feats[i].shape), smpl_feats[i]
            )
            grid_smpl_feats[i, :nr_cells, :] /= counts.unsqueeze(-1)
        out_mask[i, :nr_cells, :] = True

        if fill_value != 0:
            grid_coords[i, nr_cells:] = fill_value

    max_nr = max(max_nr)
    grid_coords = grid_coords[..., :max_nr, :]
    grid_features = grid_features[..., :max_nr, :]
    out_mask = out_mask[..., :max_nr, :]
    if smpl_feats != None:
        grid_smpl_feats = grid_smpl_feats[..., :max_nr, :]
        return grid_coords, grid_features, out_mask, grid_smpl_feats
    else:
        return grid_coords, grid_features, out_mask


def gridSampling(pcd: torch.Tensor, resolution: float):
    """Grid based downsampling. Returns the indices of the points which are closest to the grid cells.

    Args:
        pcd (torch.Tensor): [N,3] point coordinates
        resolution (float): grid resolution

    Returns:
        indices (torch.Tensor): [M] indices of the original point cloud, downsampled point cloud would be `points[indices]`
    """
    _quantization = 1000

    offset = torch.floor(pcd.min(dim=-2)[0] / resolution).long()
    grid = torch.floor(pcd / resolution)
    center = (grid + 0.5) * resolution
    dist = ((pcd - center) ** 2).sum(dim=1) ** 0.5
    dist = dist / dist.max() * (_quantization - 1)

    grid = grid.long() - offset
    v_size = grid.max().ceil()
    grid_idx = grid[:, 0] + grid[:, 1] * v_size + grid[:, 2] * v_size * v_size

    unique, inverse = torch.unique(grid_idx, return_inverse=True)
    idx_d = torch.arange(inverse.size(
        0), dtype=inverse.dtype, device=inverse.device)

    offset = 10 ** len(str(idx_d.max().item()))

    idx_d = idx_d + dist.long() * offset
    idx = torch.empty(
        unique.shape, dtype=inverse.dtype, device=inverse.device
    ).scatter_reduce_(
        dim=0, index=inverse, src=idx_d, reduce="amin", include_self=False
    )
    idx = idx % offset
    return idx


class VoxelHasherIndex(nn.Module):
    def __init__(
        self,
        points: torch.Tensor,
        grid_resolution: float,
        buffer_size=int(1e8),
        random=False,
        check_voxel_collision=False
    ) -> None:
        """Voxel Hasher for downsampling and finding neighbors in the grid. Stores in each cell the index of a point from the original point cloud

        Args:
            points (torch.Tensor): [N,3] point coordinates
            grid_resolution (float): resolution of the grid
            buffer_size (int, optional): Size of the grid buffer. The higher, the less likely voxel collisions. Defaults to 100000000.
            random (bool, optional): If True: stores random point in the cell. If False: stores the point which is closest to the voxel center.
        """
        super().__init__()
        self.primes = torch.tensor(
            [73856093, 19349669, 83492791], dtype=torch.int64, device=points.device
        )
        self.buffer_pt_index = torch.full(
            [buffer_size], -1, dtype=torch.int64, device=points.device
        )
        self.buffer_valids = torch.zeros(
            [buffer_size], dtype=torch.bool, device=points.device
        )
        self.grid_resolution = grid_resolution
        self.buffer_size = buffer_size

        # Fill grid
        if random:
            indices = torch.arange(
                points.shape[-2],
                dtype=self.buffer_pt_index.dtype,
                device=self.buffer_pt_index.device,
            )
            grid_coords = (
                points / self.grid_resolution).floor().to(self.primes)
        else:
            indices = gridSampling(points, resolution=self.grid_resolution)
            grid_coords = (
                (points[indices] / self.grid_resolution).floor().to(self.primes)
            )

        hash = (grid_coords * self.primes).sum(-1) % self.buffer_size
        self.buffer_pt_index[hash] = indices
        self.buffer_valids[hash] = True

        if check_voxel_collision:
            idc = self.neighborhood_voxel_search(
                points, num_cells=0).squeeze(-1)
            dist = (points - points[idc]).norm(dim=-1)
            collisions = (dist > (1.74*grid_resolution)).sum()
            print(f"num collisions: {collisions} in {len(points)} [{collisions/len(points):.3f} %]")

    def get_indices(self):
        """returns the indices of the points that are stored in the grid. Indices are from the original point cloud

        Returns:
            indices: torch.Tensor [K]
        """
        return self.buffer_pt_index[self.buffer_valids]

    def reset_indices(self):
        self.buffer_pt_index[self.buffer_valids] = torch.arange(
            self.buffer_valids.sum(),
            dtype=self.buffer_pt_index.dtype,
            device=self.buffer_pt_index.device,
        )

    def radius_neighborhood_search(self, points: torch.Tensor, radius: float):
        """returns the indices of the potential neighbors for each point. Be aware that those might be invalid (value: -1) or just wrong due to hash collision.

        Args:
            points [N,3] (torch.Tensor): point coordinates from which to find neighbors
            radius (float): radius in which to find neighbors, be aware that the actual search radius might be higher due to rounding up to full voxel resolution

        Returns:
            indices [N,m] (torch.Tensor): for each point the m potential neighbors. m depens of radius. For 0 < m <= voxel_resolution: 3^3 = 27 neighbors, 2*voxel_resolution: 5^3 = 125...
        """
        grid_coords = (points / self.grid_resolution).floor().to(self.primes)

        num_cells = math.ceil(radius / self.grid_resolution)
        dx = torch.arange(
            -num_cells,
            num_cells + 1,
            device=grid_coords.device,
            dtype=grid_coords.dtype,
        )

        coords = torch.meshgrid(dx, dx, dx, indexing="ij")
        dx = torch.stack(coords, dim=-1).reshape(-1, 3)

        neighbord_cells = grid_coords[..., None, :] + dx
        hash = (neighbord_cells * self.primes).sum(-1) % self.buffer_size
        return self.buffer_pt_index[hash]

    def neighborhood_voxel_search(self, points: torch.Tensor, num_cells: int = 1):
        """returns the indices of the potential neighbors for each point. Be aware that those might be invalid (value: -1) or just wrong due to hash collision.

        Args:
            points [N,3] (torch.Tensor): point coordinates from which to find neighbors
            radius (float): radius in which to find neighbors, be aware that the actual search radius might be higher due to rounding up to full voxel resolution

        Returns:
            indices [N,m] (torch.Tensor): for each point the m potential neighbors. m depens of radius. For 0 < m <= voxel_resolution: 3^3 = 27 neighbors, 2*voxel_resolution: 5^3 = 125...
        """
        grid_coords = (points / self.grid_resolution).floor().to(self.primes)

        dx = torch.arange(
            -num_cells,
            num_cells + 1,
            device=grid_coords.device,
            dtype=grid_coords.dtype,
        )

        coords = torch.meshgrid(dx, dx, dx, indexing="ij")
        dx = torch.stack(coords, dim=-1).reshape(-1, 3)

        neighbord_cells = grid_coords[..., None, :] + dx
        hash = (neighbord_cells * self.primes).sum(-1) % self.buffer_size
        return self.buffer_pt_index[hash]


def softmax(t, mask=None, dim=0, epsilon=1e-9):
    if mask is not None:
        t_exp = torch.exp(t)*mask.float()
        sm = t_exp/(torch.sum(t_exp, dim=dim, keepdim=True)+epsilon)
        return sm
    else:
        return nn.functional.softmax(t, dim)


def grid_interpolate(grid: VoxelHasherIndex, src_pts: torch.Tensor, src_feats: torch.Tensor, q_pts: torch.Tensor, num_cells=1, inv_dist=True):
    idx = grid.neighborhood_voxel_search(q_pts, num_cells)
    # N,1,3 - N,27,3 -> N,27
    dist = (q_pts[:, None, :] - src_pts[idx]).norm(dim=-1)
    if inv_dist:
        weight = 1/(dist+1e-8)
        weight = softmax(weight-weight.max(dim=-1, keepdim=True)
                         [0], mask=(idx >= 0), dim=-1)
    else:
        weight = softmax(-dist, mask=(idx >= 0), dim=-1)

    return (src_feats[idx] * weight[..., None]).sum(dim=-2)


class UpsampleBlock(nn.Module):
    def __init__(self):
        """Nearest Neighbor upsampling"""
        super().__init__()

    def forward(
        self,
        query_points: torch.Tensor,
        target_points: torch.Tensor,
        target_features: torch.Tensor,
        q_mask=None,
        t_mask=None,
    ):
        """Gets for each query point the feature of the nearest target point

        Args:
            query_points (torch.Tensor): [n x 3]
            target_points (torch.Tensor): [m x 3]
            target_features (torch.Tensor): [m x f_dim]

        Returns:
            query_points (torch.Tensor): [n x 3]
            query_features (torch.Tensor): [n x f_dim]
        """
        idx = masked_knn_keops(
            query_points, target_points, q_mask=q_mask, s_mask=t_mask, k=1
        )  # get nearest neighbor
        target_shape = list(target_features.shape)
        target_shape[-2] = idx.shape[-2]
        return query_points, torch.gather(target_features, -2, idx.expand(target_shape))
