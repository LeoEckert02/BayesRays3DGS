import torch


def find_grid_indices(points, lod, device):
    pos = normalize_point_coords(points)
    uncertainty_lod = 2**lod
    coords = (pos * uncertainty_lod).unsqueeze(0)
    inds = torch.zeros((8, pos.shape[0]), dtype=torch.int32, device=device)
    coefs = torch.zeros((8, pos.shape[0]), device=device)
    corners = torch.tensor(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 1],
            [2, 0, 1, 0],
            [3, 0, 1, 1],
            [4, 1, 0, 0],
            [5, 1, 0, 1],
            [6, 1, 1, 0],
            [7, 1, 1, 1],
        ],
        device=device,
    )
    corners = corners.unsqueeze(1)
    inds[corners[:, :, 0].squeeze(1)] = (
        (torch.floor(coords[..., 0]) + corners[:, :, 1])
        * uncertainty_lod
        * uncertainty_lod
        + (torch.floor(coords[..., 1]) + corners[:, :, 2]) * uncertainty_lod
        + (torch.floor(coords[..., 2]) + corners[:, :, 3])
    ).int()
    coefs[corners[:, :, 0].squeeze(1)] = (
        torch.abs(
            coords[..., 0] - (torch.floor(coords[..., 0]) + (1 - corners[:, :, 1]))
        )
        * torch.abs(
            coords[..., 1] - (torch.floor(coords[..., 1]) + (1 - corners[:, :, 2]))
        )
        * torch.abs(
            coords[..., 2] - (torch.floor(coords[..., 2]) + (1 - corners[:, :, 3]))
        )
    )

    return inds, coefs


def normalize_point_coords(points):
    min_val = torch.min(points)
    max_val = torch.max(points)
    return (points - min_val) / (max_val - min_val)
