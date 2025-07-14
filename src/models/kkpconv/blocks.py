from torch.functional import _return_counts
from torch.nn.init import kaiming_uniform_
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch
import math
from models.kkpconv import orig_kpconv as okp
from models.kkpconv import utils, sampling

import open3d as o3d
import numpy as np

# import pytimer
from models.kkpconv.orig_kpconv import getKernelPoints, gather


class KPUnet(nn.Module):
    def __init__(
        self,
        out_dim,
        start_voxel_size,
        init_voxel_size=None,
        end_voxel_size=None,
        in_dim=1,
        hidden_feature_dim=128,
        num_subsamplings=3,
        num_stem_layers=5,
        conv_radius_scale=1.5,
        do_deconvolution=False,
        random_sample=False,
        do_interpolation=True,
        log_voxel_scaling=True,
        aggregation_method="conv",
    ) -> None:
        super().__init__()
        self.num_stem_layers = num_stem_layers
        self.num_subsamplings = num_subsamplings
        self.random_sample = random_sample
        self.do_interpolation = do_interpolation
        self.voxel_sizes = self._compute_voxel_sizes(
            num_subsamplings, end_voxel_size, start_voxel_size, log_space=log_voxel_scaling
        )
        self.conv_radii = self.voxel_sizes * conv_radius_scale
        self.init_voxel_size = (
            start_voxel_size if init_voxel_size is None else init_voxel_size
        )
        # compute block feature dims

        min_dim = hidden_feature_dim / 4
        feature_dims = [in_dim] + torch.linspace(
            min_dim,
            hidden_feature_dim,
            steps=self.num_subsamplings,
            dtype=torch.int32,
        ).tolist()
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    in_dim=idim,
                    out_dim=odim,
                    radius=conv_r,
                    num_conv_layers=2,
                    aggregation_method=aggregation_method,
                )
                for idim, odim, conv_r in zip(
                    feature_dims[:-1], feature_dims[1:], self.conv_radii
                )
            ]
        )

        self.stem = EncoderBlock(
            hidden_feature_dim,
            hidden_feature_dim,
            radius=self.conv_radii[-1],
            num_conv_layers=num_stem_layers,
            aggregation_method=aggregation_method,
        )

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    in_dim=hidden_feature_dim,
                    out_dim=hidden_feature_dim,
                    radius=conv_r,
                    num_conv_layers=2,
                    skip_dim=skip_dim,
                    aggregation_method=aggregation_method,
                )
                for conv_r, skip_dim in zip(
                    reversed(self.conv_radii), reversed(feature_dims[1:])
                )
            ]
        )

        self.out_mlp = DecreasingMLP(
            4, hidden_feature_dim, out_dim, final_layer_norm=False
        )

    def forward(self, in_points, in_feats):
        if isinstance(in_points, torch.Tensor) and len(in_points.shape) == 2:
            return self._forward(in_points, in_feats)
        else:
            out_feats, out_points = [], []
            for p, f in zip(in_points, in_feats):
                op, of = self.forward(p, f)
                out_feats.append(of)
                out_points.append(op)
            return out_points, out_feats

    def _forward(self, in_points, in_feats):
        points_feats = [[in_points, in_feats]]
        # encoder
        grid = None
        for layer_i in range(self.num_subsamplings):
            grid = sampling.VoxelHasherIndex(
                points_feats[-1][0],
                grid_resolution=self.voxel_sizes[layer_i],
                random=self.random_sample,
                buffer_size=int(1e8) if layer_i == 0 else int(1e7),
            )
            points_feats.append(
                self.encoder_blocks[layer_i](
                    points_feats[-1][0], points_feats[-1][1], grid
                )
            )

        # stem
        grid.reset_indices()
        points_feats.append(
            self.stem(points_feats[-1][0], points_feats[-1][1], grid))

        # decoder
        for layer_i in range(len(self.decoder_blocks)):
            query_pts = points_feats[self.num_subsamplings - layer_i][0]
            if self.do_interpolation:
                query_feats = sampling.grid_interpolate(
                    grid, points_feats[-1][0], points_feats[-1][1], query_pts, num_cells=1
                )
            else:
                upsample_idx = grid.neighborhood_voxel_search(
                    query_pts, 0).squeeze(-1)
                query_feats = points_feats[-1][1][
                    upsample_idx
                ]
            skip_feats = points_feats[self.num_subsamplings - layer_i][1]
            grid = sampling.VoxelHasherIndex(
                query_pts,
                grid_resolution=list(reversed(self.voxel_sizes))[layer_i],
                random=self.random_sample,
                buffer_size=int(1e8) if layer_i == len(
                    self.decoder_blocks)-1 else int(1e7),
            )
            # assert len(query_pts) == grid.buffer_valids.sum(), f"{len(query_pts)},{grid.buffer_valids.sum()}"
            points_feats.append(
                self.decoder_blocks[layer_i](
                    query_pts, query_feats, skip_feats, grid)
            )

        out_feat = self.out_mlp(points_feats[-1][1])
        out_points = points_feats[0][0]
        if self.do_interpolation:
            out_feat = sampling.grid_interpolate(
                grid, points_feats[-1][0], out_feat, out_points, num_cells=1, inv_dist=False
            )
        else:
            upsample_idx = grid.neighborhood_voxel_search(
                out_points, 0).squeeze(-1)
            out_feat = out_feat[upsample_idx]
        return out_points, out_feat

    @staticmethod
    def _compute_voxel_sizes(num_subsamplings, end_voxel_size, start_voxel_size, log_space=True):
        if log_space:
            voxel_sizes = (
                torch.logspace(
                    start=0,
                    end=num_subsamplings - 1,
                    steps=num_subsamplings,
                    base=2
                    if end_voxel_size is None
                    else (end_voxel_size / start_voxel_size)
                    ** (1 / (num_subsamplings - 1)),
                )
                * start_voxel_size
            ).numpy()
        else:
            # linear space
            voxel_sizes = torch.linspace(
                start_voxel_size, end_voxel_size, num_subsamplings).numpy()

        return voxel_sizes


class EncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, radius, num_conv_layers=1, aggregation_method="conv"):
        super().__init__()
        if num_conv_layers >= 1:
            block = ResnetKPConv if aggregation_method == "conv" else LocalPointAttention
            self.blocks = nn.ModuleList(
                [block(in_dim, out_dim, radius=radius)])
            for _ in range(num_conv_layers - 1):
                self.blocks.append(block(
                    out_dim, out_dim, radius=radius))
        else:
            self.blocks = nn.ModuleList(nn.Identity())

    def forward(self, src_pts, src_feats, grid: sampling.VoxelHasherIndex):
        for i, block in enumerate(self.blocks):
            if i == 0:
                query_idx = grid.get_indices()
                query_pts = src_pts[query_idx]
                query_feats = src_feats[query_idx]
                neighb_idx = grid.neighborhood_voxel_search(query_pts, 1)
            elif i == 1:
                src_pts = query_pts
                src_feats = query_feats
                grid.reset_indices()
                neighb_idx = grid.neighborhood_voxel_search(query_pts, 1)
            else:
                src_feats = query_feats

            query_feats = block(
                query_pts,
                src_pts,
                neighb_idx,
                src_feats=src_feats,
                query_feats=query_feats,
            )
        return query_pts, query_feats


class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, radius, skip_dim, num_conv_layers=1, aggregation_method="conv"):
        super().__init__()
        if num_conv_layers >= 1:
            block = ResnetKPConv if aggregation_method == "conv" else LocalPointAttention
            self.blocks = nn.ModuleList(
                [block(in_dim, out_dim,
                              skip_dim=skip_dim, radius=radius)]
            )
            for _ in range(num_conv_layers - 1):
                self.blocks.append(block(
                    out_dim, out_dim, radius=radius))
        else:
            self.blocks = nn.ModuleList(nn.Identity())

    def forward(self, pts, feats, skip_feats, grid: sampling.VoxelHasherIndex):

        for i, block in enumerate(self.blocks):
            if i == 0:
                neighb_idx = grid.neighborhood_voxel_search(pts, 1)

            feats = block(
                pts,
                pts,
                neighb_idx,
                src_feats=feats,
                query_feats=skip_feats,
            )
            skip_feats = feats
        return pts, feats


class DecreasingMLP(nn.Module):
    """
    Multilayer Perceptron.
    """

    def __init__(self, n_layers, in_feat_size, out_feat_size, final_layer_norm=False):
        super().__init__()

        size_step = (in_feat_size - out_feat_size) / (n_layers)
        layer_list = []

        for l in range(n_layers):
            layer_in_size = int(in_feat_size - l * size_step)
            layer_out_size = int(in_feat_size - (l + 1) * size_step)
            layer_list.append(nn.Linear(layer_in_size, layer_out_size))
            if l < (n_layers - 1):
                layer_list.append(nn.LeakyReLU())
            if (l < (n_layers - 1)) or final_layer_norm:
                layer_list.append(nn.LayerNorm(layer_out_size))
        self.layers = nn.Sequential(*layer_list)
        print("MLP", self.layers)

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)


class ResnetKPConv(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        radius,
        kernel_size=3,
        p_dim=3,
        f_dscale=2,
        layernorm=False,
        skip_dim=None,
    ):
        super().__init__()
        skip_dim = skip_dim if skip_dim is not None else in_dim
        self.ln1 = nn.LayerNorm(in_dim) if layernorm else nn.Identity()
        self.relu = nn.LeakyReLU()

        self.kpconv = KPConv(
            in_dim=in_dim,
            out_dim=out_dim // f_dscale,
            radius=radius,
            kernel_size=kernel_size,
            p_dim=p_dim,
        )

        self.ln2 = nn.LayerNorm(
            out_dim // f_dscale) if layernorm else nn.Identity()
        self.lin = nn.Linear(out_dim // f_dscale, out_dim)

        self.in_projection = (
            nn.Identity() if skip_dim == out_dim else nn.Linear(skip_dim, out_dim)
        )

    def forward(self, q_pts, s_pts, neighb_inds, src_feats, query_feats):
        xr = self.relu(self.ln1(src_feats))
        xr = self.kpconv(q_pts, s_pts, neighb_inds, src_feats)
        xr = self.relu(self.ln2(xr))
        xr = self.lin(xr)

        return self.in_projection(query_feats) + xr


class GridSampleConv(nn.Module):
    def __init__(
        self,
        in_fdim,
        out_fdim,
        subsampling_dist,
        kernel_radius,
        num_kernel_points=3,
        num_neighbors=32,
        preactivate=True,
        layernorm=False,
        relu=True,
        kernel_debug_viz=False,
        grid_sampling_method="mean",
    ):
        """KPConv resnet Block with downsampling

        Args:
            in_fdim (int): feature dimension of input
            out_fdim (int): feature dimension of output
            subsampling_dist (float): resolution of the grid to subsample, no subsampling if <=0
            kernel_radius (float): radius of the convolutional kernel
            num_kernel_points (int, optional): num kernel points in each dimension of a grid(e.g 3=> 3^3 =27). Defaults to 3.
            num_neighbors (int, optional): Number neighbors for the convolution. Defaults to 32.
            preactivate (bool, optional): bool if preactivate. Don't do this if std(feature) is 0. Defaults to True.
            layernorm (bool, optional): bool if use layernorm. Defaults to True.
            deformable (bool, optional): bool if using deformable kpconv. Defaults to False.
            relu (bool, optional): bool if use relu. Defaults to True.
            kernel_debug_viz: visualize the input, query and kernel points for debugging and param tuning
        """
        super().__init__()
        self.in_fdim = in_fdim
        self.out_fdim = out_fdim
        conf_in_fdim = out_fdim if preactivate else in_fdim
        self.subsampling_dist = subsampling_dist
        self.kernel_radius = kernel_radius
        self.num_neighbors = num_neighbors
        self.grid_sampling_method = grid_sampling_method

        ### Preactivation ####
        self.relu = nn.LeakyReLU()
        if preactivate:
            pre_blocks = [
                nn.Linear(in_features=in_fdim, out_features=out_fdim)]
            if layernorm:
                pre_blocks.append(BitchNorm(out_fdim))
            if relu:
                pre_blocks.append(self.relu)
            self.preactivation = nn.ModuleList(pre_blocks)
        else:
            self.preactivation = nn.ModuleList(
                [
                    nn.Identity(),
                ]
            )
        # KP Conv
        self.kp_conv = KPConv(
            kernel_size=num_kernel_points,
            p_dim=3,
            in_dim=conf_in_fdim,
            out_dim=out_fdim,
            radius=self.kernel_radius,
            kernel_debug_viz=kernel_debug_viz,
        )

        # print('kernel radius', self.kernel_radius)
        # Post linear
        post_layer = []
        if layernorm:
            post_layer.append(BitchNorm(out_fdim))
        if relu:
            post_layer.append(self.relu)
        post_layer.append(
            nn.Linear(in_features=out_fdim, out_features=out_fdim))
        if layernorm:
            post_layer.append(BitchNorm(out_fdim))
        self.post_layer = nn.ModuleList(post_layer)

        # Shortcut
        self.shortcut = nn.ModuleList(
            [
                nn.Identity(),
            ]
        )
        if in_fdim != out_fdim:
            sc_blocks = [nn.Linear(in_features=in_fdim, out_features=out_fdim)]
            if layernorm:
                sc_blocks.append(BitchNorm(out_fdim))
            self.shortcut = nn.ModuleList(sc_blocks)

    def apply_module_list(self, module_list, features, mask):
        for block in module_list:
            if isinstance(block, BitchNorm):
                features = block(features, mask)
            else:
                features = block(features)
        return features

    def forward(
        self,
        src_points: torch.Tensor,
        src_features: torch.Tensor,
        query_points=None,
        query_features=None,
        return_smpl_ids=False,
    ):
        """Computes a convolution for a subsampled subset of the input src_points

        Args:
            src_points (Tensor): [n x 3]
            features (Tensor): [n x in_fdim]
            smpl_feats (Tensor): [n x in_fdim] additional point features that get only grid sampled

        Returns:
            src_points: [m x 3], m <= n
            features: [m x out_fdim]
            sampled_smpl_feats: [m x out_fdim] additional point features that have been only grid sampled

        """
        # print("subsmpl dist", self.subsampling_dist)

        query_points_out = []
        out_features_out = []
        query_mask_out = []
        neighbors_index_out = []

        for B in range(len(src_points)):
            src_points_batch = src_points[B].unsqueeze(0)
            src_features_batch = src_features[B].unsqueeze(0)
            if query_points != None:
                query_points_batch = query_points[B].unsqueeze(0)
            # src_mask_batch = src_mask[B] if src_mask is not None else None

            if query_points == None:
                (
                    query_points_batch,
                    query_features_batch,
                    neighbors_index_batch,
                ) = hashed_sampling_knn(
                    src_points_batch,
                    features=src_features_batch,
                    search_dist=self.subsampling_dist,
                    resolution=self.subsampling_dist,
                    fill_value=1e6,
                    random=False,
                    do_sampling=self.subsampling_dist > 0,
                )
            else:
                neighbors_index_batch = hashed_knn(
                    src_pcd=src_points_batch,
                    query_pcd=query_points_batch,
                    # src_mask=src_mask_batch,
                    resolution=self.subsampling_dist,
                    search_dist=self.subsampling_dist,
                    random=False,
                ).unsqueeze(0)

                if query_features == None:
                    s_pts = torch.cat(
                        (
                            src_points_batch,
                            torch.zeros_like(
                                src_points_batch[..., :1, :]) + 1e6,
                        ),
                        -2,
                    )
                    # x = torch.cat((src_features, torch.zeros_like(src_features[..., :1, :])), -2)
                    neighbors_index_batch[neighbors_index_batch == -1] = (
                        s_pts.shape[1] - 1
                    )
                    neighbors = utils.vector_gather(
                        s_pts, neighbors_index_batch)
                    closest_ids = torch.norm(
                        query_points_batch.unsqueeze(-2).expand_as(neighbors)
                        - neighbors,
                        dim=-1,
                    ).argmin(dim=-1)
                    query_features_batch = utils.vector_gather(
                        src_features_batch, closest_ids.unsqueeze(-1)
                    )
                    query_features_batch = query_features_batch.squeeze(-2)
                else:
                    query_features_batch = query_features

            out_features_batch = self.apply_module_list(
                self.preactivation, src_features_batch, None
            )
            # print(query_points_batch.shape, src_points_batch.shape, neighbors_index_batch.shape, out_features_batch.shape)
            out_features_batch = self.kp_conv.forward(
                q_pts=query_points_batch,
                s_pts=src_points_batch,
                neighb_inds=neighbors_index_batch,
                x=out_features_batch,
            )

            out_features_batch = self.apply_module_list(
                self.post_layer, out_features_batch, None
            )
            try:
                out_features_batch = self.relu(
                    self.apply_module_list(
                        self.shortcut, query_features_batch, None)
                    + out_features_batch
                )
            except:
                import ipdb

                ipdb.set_trace()  # fmt: skip

            query_points_out.append(query_points_batch[0])
            out_features_out.append(out_features_batch[0])
            query_mask_out.append(None)
            neighbors_index_out.append(neighbors_index_batch[0])

        return_vals = [query_points_out, out_features_out, query_mask_out]
        if return_smpl_ids:
            return_vals.append(neighbors_index_out)
        # if smpl_feats != None:
        #     return_vals.append(sampling_return[-1])
        # if out_features.isnan().any():
        #     import ipdb;ipdb.set_trace()  # fmt: skip
        return return_vals


##############################################################
# KKPConv
##############################################################


class KPConv(nn.Module):
    def __init__(
        self,
        kernel_size,
        in_dim,
        out_dim,
        radius,
        KP_extent=None,
        p_dim=3,
        kernel_debug_viz=False,
    ):
        """
        Initialize parameters for KPConvDeformable.
        :param kernel_size: Number of kernel points.
        :param in_dim: dimension of input features.
        :param out_dim: dimension of output features.
        :param radius: radius used for kernel point init.
        :param KP_extent: influence radius of each kernel point. (float), default: None
        :param p_dim: dimension of the point space. Default: 3
        :param radial: bool if direction independend convolution
        :param align_kp: aligns the kernel points along the main directions of the local neighborhood
        :param kernel_debug_viz: visualize the input, query and kernel points for debugging and param tuning
        """
        super(KPConv, self).__init__()

        # Save parameters
        self.p_dim = p_dim

        self.K = kernel_size**self.p_dim
        self.num_kernels = kernel_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.radius = radius
        self.KP_extent = (
            2 * radius / (kernel_size - 1) / self.p_dim**0.5
            if KP_extent is None
            else KP_extent
        )
        self.kernel_debug_viz = kernel_debug_viz

        # Initialize weights
        self.weights = Parameter(
            torch.zeros((self.K, in_dim, out_dim), dtype=torch.float32),
            requires_grad=True,
        )

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        self.kernel_points = self.init_KP()

        return

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        return

    def init_KP(self):
        """
        Initialize the kernel point positions in a grid
        :return: the tensor of kernel points
        """
        K_points_numpy = getKernelPoints(
            self.radius, self.num_kernels, dim=self.p_dim)
        return Parameter(
            torch.tensor(K_points_numpy, dtype=torch.float32), requires_grad=False
        )

    def visualize_pcl_and_kernels(self, q_pts, s_pts, neighbors):
        print(
            "Shapes q:{} s:{} neigh:{}".format(
                q_pts.shape, s_pts.shape, neighbors.shape
            )
        )
        src_cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(
                s_pts[(s_pts != 1e6).all(-1)].detach().cpu().numpy()
            )
        )
        src_cloud.paint_uniform_color(np.array((1.0, 0, 0)))

        neighb_cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(
                neighbors[::100][(neighbors[::100] != 1e6).all(-1)]
                .detach()
                .cpu()
                .numpy()
            )
        )
        neighb_cloud.paint_uniform_color(np.array((1.0, 0, 1.0)))
        neighb_cloud.translate(np.array((-0.0002, -0.0002, -0.0002)))

        query_cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(
                q_pts[(q_pts != 1e6).all(-1)].detach().cpu().numpy()
            )
        )
        query_cloud.paint_uniform_color(np.array((0, 1.0, 0)))
        query_cloud.translate(np.array((0.0002, 0.0002, 0.0002)))
        kernel_points = q_pts[(q_pts != 1e6).all(-1)][::100].unsqueeze(
            -2
        ) + self.kernel_points.unsqueeze(-3)
        shape = kernel_points.shape
        kernel_cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(
                kernel_points.reshape(shape[0] * shape[1], shape[2])
                .detach()
                .cpu()
                .numpy()
            )
        )
        kernel_cloud.paint_uniform_color(np.array((0, 0, 1.0)))
        o3d.visualization.draw_geometries(
            [query_cloud, kernel_cloud, neighb_cloud])

    def forward(self, q_pts, s_pts, neighb_inds, x):
        # Add a fake point/feature in the last row for shadow neighbors
        s_pts = torch.cat(
            (s_pts, torch.zeros_like(s_pts[..., :1, :]) + 1e6), -2)
        x = torch.cat((x, torch.zeros_like(x[..., :1, :])), -2)
        neighb_inds = neighb_inds % s_pts.shape[-2]

        # print(q_pts.shape, s_pts.shape, neighb_inds.shape, x.shape)
        # Get neighbor points and features [n_points, n_neighbors, dim/ in_fdim]
        if len(neighb_inds.shape) < 3:
            neighbors = s_pts[neighb_inds, :]
            neighb_x = gather(x, neighb_inds)
        else:
            neighbors = utils.vector_gather(s_pts, neighb_inds)
            neighb_x = utils.vector_gather(x, neighb_inds)
        if self.kernel_debug_viz:
            self.visualize_pcl_and_kernels(q_pts, s_pts, neighbors)

        if len(neighbors.shape) > 3:
            import ipdb

            ipdb.set_trace()  # fmt: skip
        # Center every neighborhood [n_points, n_neighbors, dim]
        neighbors = neighbors - q_pts.unsqueeze(-2)

        # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
        neighbors.unsqueeze_(-2)
        differences = neighbors - self.kernel_points
        # Get the square distances [n_points, n_neighbors, n_kpoints]
        sq_distances = torch.sum(differences**2, dim=-1)
        # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
        all_weights = torch.clamp(
            1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0
        )
        fx = torch.einsum(
            "...nkl,...nki,...lio->...no", all_weights, neighb_x, self.weights
        )
        return fx

    def __repr__(self):
        return "KPConv(radius: {:.2f}, in_feat: {:d}, out_feat: {:d})".format(
            self.radius, self.in_dim, self.out_dim
        )


class PositionalEncoder(nn.Module):
    # out_dim = in_dimnesionality * (2 * bands + 1)
    def __init__(self, freq, num_bands=5, dimensionality=3, base=2):
        super().__init__()
        self.freq, self.num_bands = torch.tensor(freq), num_bands
        self.dimensionality, self.base = dimensionality, torch.tensor(base)
        # self.num_bands = floor(feature_size/dimensionality/2)

    def forward(self, x):
        x = x[..., :self.dimensionality, None]
        device, dtype, orig_x = x.device, x.dtype, x

        scales = torch.logspace(0., torch.log(
            self.freq / 2) / torch.log(self.base), self.num_bands, base=self.base, device=device, dtype=dtype)
        # Fancy reshaping
        scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

        x = x * scales * math.pi
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = torch.cat((x, orig_x), dim=-1)
        x = x.flatten(-2, -1)
        return x

    def featureSize(self):
        return self.dimensionality*(self.num_bands*2+1)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim-in_dim)

    def forward(self, x):
        return torch.cat([x, self.lin(2 * math.pi * x).cos()], dim=-1)


class LocalPointAttention(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        radius,
        skip_dim=None,
        p_dim=3,
    ):
        super().__init__()
        self.radius = radius
        self.cross_attention_block = nn.TransformerDecoderLayer(d_model=out_dim,
                                                                nhead=out_dim//16,  # hope its good
                                                                dim_feedforward=out_dim,
                                                                dropout=0,
                                                                batch_first=True,
                                                                norm_first=True)  # check this: https://arxiv.org/pdf/2002.04745.pdf
        self.pos_encoding = LearnablePositionalEncoding(
            in_dim=p_dim,
            out_dim=out_dim)

        self.proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        if skip_dim is None:
            self.skip_proj = self.proj
        else:
            self.skip_proj = nn.Identity() if skip_dim == out_dim else nn.Linear(skip_dim, out_dim)
                      
    def forward(self, q_pts, s_pts, neighb_inds, src_feats, query_feats, invalid_neigbhor_mask=None):
        s_pts = torch.cat(
            (s_pts, torch.zeros_like(s_pts[..., :1, :]) + 1e6), -2)
        src_feats = torch.cat(
            (src_feats, torch.zeros_like(src_feats[..., :1, :])), -2)
        src_feats = self.proj(src_feats)  # proj
        query_feats = self.skip_proj(query_feats)

        neighb_inds = neighb_inds % s_pts.shape[-2]

        # Get neighbor points and features [n_points, n_neighbors, dim/ in_fdim]
        if len(neighb_inds.shape) < 3:
            neighbors = s_pts[neighb_inds, :]
            neighb_x = gather(src_feats, neighb_inds)
        else:
            neighbors = utils.vector_gather(s_pts, neighb_inds)
            neighb_x = utils.vector_gather(src_feats, neighb_inds)

        # Center every neighborhood [n_points, n_neighbors, dim]
        neighbors = neighbors - q_pts.unsqueeze(-2)
        neighb_x = neighb_x + self.pos_encoding(neighbors/self.radius)

        query_x = query_feats + self.pos_encoding(torch.zeros_like(q_pts))

        if invalid_neigbhor_mask is None:
            invalid_neigbhor_mask = neighbors.norm(dim=-1) > self.radius
        out = self.cross_attention_block(tgt=query_x[..., None, :], memory=neighb_x,
                                         memory_key_padding_mask=invalid_neigbhor_mask).squeeze(-2)
        if out.isnan().any():
            out[out.isnan()] = 0
            # import ipdb;ipdb.set_trace()  # fmt: skip
        return out


def sumByIdx(target_shape, indices, src):
    sum = torch.zeros(target_shape, dtype=src.dtype, device=src.device)
    sum.scatter_add_(-2, indices.expand(src.shape), src)
    return sum


def getShape(shape, val, idx):
    b = list(shape)
    b[idx] = val
    return b


def hashed_knn(
    src_pcd: torch.Tensor,
    query_pcd: torch.Tensor,
    #    src_mask: torch.Tensor,
    resolution: float,
    search_dist: float,
    random=False,
):
    if len(src_pcd.shape) < 3:
        src_pcd = src_pcd.unsqueeze(0)

    # if src_mask is not None:
    #     src_pcd[~src_mask.expand_as(src_pcd)] = float("inf")

    B = src_pcd.shape[0]
    neighb_size = (math.ceil(search_dist / resolution) * 2 + 1) ** 3
    neighbors_index = torch.full(
        (B, max(src_pcd.shape[1], query_pcd.shape[1]), neighb_size),
        -1,
        dtype=torch.int64,
        device=src_pcd.device,
    )
    for i in range(B):
        hasher = sampling.VoxelHasherIndex(
            src_pcd[i], resolution, buffer_size=int(1e8), random=random
        )
        neighb_idx = hasher.neighborhood_voxel_search(
            query_pcd[i], search_dist)
        neighbors_index[i, : len(neighb_idx), :] = neighb_idx
    return neighbors_index[0]


def hashed_sampling_knn(
    pcd: torch.Tensor,
    resolution: float,
    search_dist: float,
    features=None,
    random=False,
    do_sampling=True,
):
    if len(pcd.shape) < 3:
        pcd = pcd.unsqueeze(0)
    if len(features.shape) < 3:
        features = features.unsqueeze(0)
    B = pcd.shape[0]
    if do_sampling:
        downsampled_pcd = torch.zeros_like(pcd, device=pcd.device)
        downsampled_features = torch.zeros_like(features, device=pcd.device)

        # out_mask = torch.full_like(pcd[..., :1], False, dtype=bool, device=pcd.device)

        # if mask is not None:
        #     pcd[~mask.expand_as(pcd)] = float("inf")

        # if mask is not None:
        #     pcd[~mask.expand_as(pcd)] = fill_value

        max_nr = []
    neighb_size = (math.ceil(search_dist / resolution) * 2 + 1) ** 3
    neighbors_index = torch.full(
        (B, pcd.shape[1], neighb_size), -1, dtype=torch.int64, device=pcd.device
    )

    for i in range(B):
        hasher = sampling.VoxelHasherIndex(
            pcd[i], resolution, buffer_size=int(1e8), random=random
        )

        if do_sampling:
            # compute the ids of the downsampled pcd in the input pcd
            sparse_ids = hasher.get_indices()
            nr_cells = len(sparse_ids)
            max_nr.append(nr_cells)
            downsampled_batch = pcd[i, sparse_ids]
            downsampled_pcd[i, :nr_cells, :] = downsampled_batch
            downsampled_features[i, :nr_cells, :] = features[i, sparse_ids]

            # out_mask[i, :nr_cells, :] = True

            neighb_idx = hasher.neighborhood_voxel_search(
                downsampled_batch, search_dist
            )
        else:
            neighb_idx = hasher.neighborhood_voxel_search(pcd[i], search_dist)
        neighbors_index[i, : len(neighb_idx), :] = neighb_idx

    if do_sampling:
        max_nr = max(max_nr)
        downsampled_pcd = downsampled_pcd[..., :max_nr, :]
        downsampled_features = downsampled_features[..., :max_nr, :]
        # out_mask = out_mask[..., :max_nr, :]
        neighbors_index = neighbors_index[..., :max_nr, :]
    else:
        downsampled_pcd = pcd
        downsampled_features = features
        # out_mask = mask
    return downsampled_pcd, downsampled_features, neighbors_index


def randomGridSampling(
    pcd: torch.Tensor,
    resolution_meter,
    scale=1.0,
    features=None,
    smpl_feats=None,
    mask=None,
    fill_value=0,
):
    """Computes a rondom point over all points in the grid cells

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
    grid = torch.floor((pcd - pcd.min(dim=-2, keepdim=True)[0]) / resolution)

    if mask is not None:
        pcd[~mask.expand_as(pcd)] = fill_value

    if mask is not None:
        grid_size = grid[mask.squeeze(-1)].max().detach()
    else:
        grid_size = grid.max().detach()
    grid_idx = (
        grid[..., 0] + grid[..., 1] * grid_size +
        grid[..., 2] * grid_size * grid_size
    )

    max_nr = []
    for i in range(B):
        unique, indices, counts = torch.unique(
            grid_idx[i], return_inverse=True, dim=None, return_counts=True
        )

        nr_cells = len(counts)
        if unique[-1].isinf():
            counts = counts[:-1]
            nr_cells -= 1
        max_nr.append(nr_cells)
        indices.detach_()
        grid_point_ids = torch.full(
            pcd.shape[-2:-1], -1, device=pcd.device, dtype=torch.long
        )
        grid_point_ids.scatter_(
            -1, indices, torch.arange(len(indices),
                                      device=grid_point_ids.device)
        )
        grid_point_ids = grid_point_ids[:nr_cells].detach()

        grid_coords[i, :nr_cells, :] = pcd[i, grid_point_ids]

        grid_features[i, :nr_cells, :] = features[i, grid_point_ids]
        if smpl_feats != None:
            grid_smpl_feats[i, :nr_cells,
                            :] = smpl_feats[i][grid_point_ids[:nr_cells]]

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


class BitchNorm(nn.BatchNorm1d):
    def forward(self, input, valid_mask=None):
        """Computes bitch norm...

        Args:
            input (...,num_features): _description_
            valid_mask (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if valid_mask is None:
            shape = input.shape
            input = input.reshape(-1, shape[-1])
            return super().forward(input).reshape(shape)
        else:
            input[valid_mask.squeeze(-1)] = super().forward(
                input[valid_mask.squeeze(-1)]
            )
            return input
