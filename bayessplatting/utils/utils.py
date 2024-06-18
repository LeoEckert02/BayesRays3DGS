import torch
from jaxtyping import Float
from nerfstudio.data.scene_box import OrientedBox
from torch import Tensor


def find_grid_indices(points, box: OrientedBox, lod, device):
    pos = get_normalized_positions(box, points)
    pos = pos.view(-1, 3)
    uncertainty_lod = 2 ** lod
    coords = (pos * uncertainty_lod).unsqueeze(0)
    inds = torch.zeros((8, pos.shape[0]), dtype=torch.int32, device=device)
    coefs = torch.zeros((8, pos.shape[0]), device=device)
    # represent corners of grid cell, format [index, x_offset, y_offset, z_offset]
    corners = torch.tensor(
        [[0, 0, 0, 0], [1, 0, 0, 1], [2, 0, 1, 0], [3, 0, 1, 1], [4, 1, 0, 0], [5, 1, 0, 1], [6, 1, 1, 0],
         [7, 1, 1, 1]], device=device)
    corners = corners.unsqueeze(1)
    # compute the grid cell indices
    inds[corners[:, :, 0].squeeze(1)] = (
            (torch.floor(coords[..., 0]) + corners[:, :, 1]) * uncertainty_lod * uncertainty_lod +
            (torch.floor(coords[..., 1]) + corners[:, :, 2]) * uncertainty_lod +
            (torch.floor(coords[..., 2]) + corners[:, :, 3])).int()
    # compute the interpolation coefficient of every point in grid
    coefs[corners[:, :, 0].squeeze(1)] = torch.abs(
        coords[..., 0] - (torch.floor(coords[..., 0]) + (1 - corners[:, :, 1]))) * torch.abs(
        coords[..., 1] - (torch.floor(coords[..., 1]) + (1 - corners[:, :, 2]))) * torch.abs(
        coords[..., 2] - (torch.floor(coords[..., 2]) + (1 - corners[:, :, 3])))

    return inds, coefs


def get_normalized_positions(box: OrientedBox, positions: Float[Tensor, "n 3"]):
    """Return normalized positions in range [0, 1] based on the OrientedBox.

    Args:
        :param positions: the xyz positions
        :param box: the box of the scene
    """
    # Transform positions to the OrientedBox coordinate system
    r, t, s = box.R, box.T, box.S.to(positions)
    h = torch.eye(4, device=positions.device, dtype=positions.dtype)
    h[:3, :3] = r
    h[:3, 3] = t
    h_world2bbox = torch.inverse(h)
    positions = torch.cat((positions, torch.ones_like(positions[..., :1])), dim=-1)
    positions = torch.matmul(h_world2bbox, positions.T).T[..., :3]

    # Normalize positions within the OrientedBox
    # lower bound
    comp_l = torch.tensor(-s / 2)
    # upper bound
    comp_m = torch.tensor(s / 2)
    aabb_lengths = comp_m - comp_l
    normalized_positions = (positions - comp_l) / aabb_lengths
    return normalized_positions
