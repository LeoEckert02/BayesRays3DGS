import torch

from bayessplatting.utils.utils import find_grid_indices


def get_uncertainty(self, points):
    aabb = self.scene_box.aabb.to(points.device)
    # samples outside aabb will have 0 coeff and hence 0 uncertainty. To avoid problems with these samples we set
    # zero_out=False
    inds, coeffs = find_grid_indices(points, self.lod, points.device)
    cfs_2 = (coeffs ** 2) / torch.sum((coeffs ** 2), dim=0, keepdim=True)
    uns = self.un[inds.long()]  # [8,N]
    un_points = torch.sqrt(torch.sum((uns * cfs_2), dim=0)).unsqueeze(1)

    # for stability in volume rendering we use log uncertainty
    un_points = torch.log10(un_points + 1e-12)
    # un_points = un_points.view((points.shape[0], points.shape[1], 1))
    return un_points




