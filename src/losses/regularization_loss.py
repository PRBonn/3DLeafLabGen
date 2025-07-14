# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

def mesh_edge_loss(out, edges, target_length: float = 0.0):
    """
    Computes mesh edge length regularization loss averaged across all meshes
    in a batch. Each mesh contributes equally to the final loss, regardless of
    the number of edges per mesh in the batch by weighting each mesh with the
    inverse number of edges. For example, if mesh 3 (out of N) has only E=4
    edges, then the loss for each edge in mesh 3 should be multiplied by 1/E to
    contribute to the final loss.

    Args:
        out: a batch of predictions [ batch_idx , N points, 3 ].
        adjacency: edges between points [ batch_idx , N edges, 2 ].
        target_length: Resting value for the edge length.

    Returns:
        loss: Average loss across the batch. Returns 0 if meshes contains
        no meshes or all empty meshes.
    """
    if len(out) == 0:
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(out)
    edges_packed = torch.vstack(tuple(edges))
    verts_packed = torch.vstack(tuple(out))
    edge_to_mesh_idx = torch.vstack([ torch.ones(edges[x].shape[-2],1)*x for x in range(N) ])
    num_edges_per_mesh = torch.tensor([ edges[x].shape[-2] for x in range(N) ])  # N

    # Determine the weight for each edge based on the number of edges in the
    # mesh it corresponds to.
    # weights = num_edges_per_mesh.gather(0, edge_to_mesh_idx)
    weights = 1.0 / num_edges_per_mesh.float()

    verts_edges = verts_packed[edges_packed.type(torch.int)]
    v0, v1 = verts_edges.unbind(1)
    loss = ((v0 - v1).norm(dim=1, p=2) - target_length) ** 2.0
    loss = loss * weights[ edge_to_mesh_idx.type(torch.int)][:,0].cuda()

    return loss.sum() / N


def mesh_laplacian_smoothing(out, edges, faces, method: str = "uniform"):
    r"""
    Computes the laplacian smoothing objective for a batch of meshes.
    This function supports three variants of Laplacian smoothing,
    namely with uniform weights("uniform"), with cotangent weights ("cot"),
    and cotangent curvature ("cotcurv").For more details read [1, 2].

    Args:
        meshes: Meshes object with a batch of meshes.
        method: str specifying the method for the laplacian.
    Returns:
        loss: Average laplacian smoothing loss across the batch.
        Returns 0 if meshes contains no meshes or all empty meshes.

    [1] Desbrun et al, "Implicit fairing of irregular meshes using diffusion
    and curvature flow", SIGGRAPH 1999.

    [2] Nealan et al, "Laplacian Mesh Optimization", Graphite 2006.
    """
    if len(out) == 0:
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(out)
    verts_packed = torch.vstack(tuple(out))
    edges_packed = torch.vstack(tuple(edges))
    edge_to_mesh_idx = torch.vstack([ torch.ones(edges[x].shape[-2],1)*x for x in range(N) ])
    num_verts_per_mesh = torch.tensor([ out[x].shape[-2] for x in range(N) ])  # N
    verts_packed_idx = torch.vstack([ torch.ones(out[x].shape[-2],1)*x for x in range(N) ])
    weights = 1.0 / num_verts_per_mesh.float()

    #faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    faces_packed = torch.vstack(tuple(faces))

    # We don't want to backprop through the computation of the Laplacian;
    # just treat it as a magic constant matrix that is used to transform
    # verts into normals
    with torch.no_grad():
        if method == "uniform":
            from pytorch3d.ops import laplacian
            L = laplacian(verts_packed, edges_packed.type(torch.int64))
        elif method in ["cot", "cotcurv"]:
            L, inv_areas = cot_laplacian(verts_packed, faces_packed)
            if method == "cot":
                norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                idx = norm_w > 0
                # pyre-fixme[58]: `/` is not supported for operand types `float` and
                #  `Tensor`.
                norm_w[idx] = 1.0 / norm_w[idx]
            else:
                L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                norm_w = 0.25 * inv_areas
        else:
            raise ValueError("Method should be one of {uniform, cot, cotcurv}")

    if method == "uniform":
        loss = L.mm(verts_packed)
    elif method == "cot":
        # pyre-fixme[61]: `norm_w` is undefined, or not always defined.
        loss = L.mm(verts_packed) * norm_w - verts_packed
    elif method == "cotcurv":
        # pyre-fixme[61]: `norm_w` may not be initialized here.
        loss = (L.mm(verts_packed) - L_sum * verts_packed) * norm_w
    loss = loss.norm(dim=1)
    loss = loss * weights[ verts_packed_idx[:,0].type(torch.int) ].cuda()
    return loss.sum() / N
