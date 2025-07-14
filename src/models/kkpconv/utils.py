import torch



def vector_gather(vectors: torch.Tensor, indices: torch.Tensor):
    """
    Gathers (batched) vectors according to indices.
    Arguments:
        vectors: Tensor[B, N1, D]
        indices: Tensor[B, N2, K]
    Returns:
        Tensor[B,N2, K, D]
    """
    # out = vectors.transpose(0, 1)[indices, 0]
    # out = torch.gather(vectors,dim=-2,index= indices)

    # Out
    shape = list(indices.shape) + [vectors.shape[-1]]
    out = torch.zeros(shape, device=vectors.device)

    # src
    vectors = vectors.unsqueeze(-2)
    shape = list(vectors.shape)
    shape[-2] = indices.shape[-1]
    vectors = vectors.expand(shape)

    # Do the magic
    indices = indices.unsqueeze(-1).expand_as(out)
    out = torch.gather(vectors, dim=-3, index=indices)
    return out

