import torch
from gsplat import spherical_harmonics
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.model_components import renderers

from bayessplatting.utils.utils import find_grid_indices
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians


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


def get_outputs(self, camera: Cameras):
    if not isinstance(camera, Cameras):
        print("Called get_outputs with not a camera")
        return {}
    assert camera.shape[0] == 1, "Only one camera at a time"

    # get the background color
    if self.training:
        optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)[0, ...]

        if self.config.background_color == "random":
            background = torch.rand(3, device=self.device)
        elif self.config.background_color == "white":
            background = torch.ones(3, device=self.device)
        elif self.config.background_color == "black":
            background = torch.zeros(3, device=self.device)
        else:
            background = self.background_color.to(self.device)
    else:
        optimized_camera_to_world = camera.camera_to_worlds[0, ...]

        if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
            background = renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device)
        else:
            background = self.background_color.to(self.device)

    N = self.N
    reg_lambda = 1e-4 / ((2 ** self.lod) ** 3)
    H = self.hessian / N + reg_lambda
    self.un = 1 / H

    max_uncertainty = 6  # approximate upper bound of the function log10(1/(x+lambda)) when lambda=1e-4/(256^3) and x is the hessian
    min_uncertainty = -3  # approximate lower bound of that function (cutting off at hessian = 1000)

    if self.crop_box is not None and not self.training:
        crop_ids = self.crop_box.within(self.means).squeeze()
        if crop_ids.sum() == 0:
            return self.get_empty_outputs(int(camera.width.item()), int(camera.height.item()), background)
    else:
        crop_ids = None
    camera_downscale = self._get_downscale_factor()
    camera.rescale_output_resolution(1 / camera_downscale)
    # shift the camera to center of scene looking at center
    R = optimized_camera_to_world[:3, :3]  # 3 x 3
    T = optimized_camera_to_world[:3, 3:4]  # 3 x 1

    # flip the z and y axes to align with gsplat conventions
    R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
    R = R @ R_edit
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.T
    T_inv = -R_inv @ T
    viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
    viewmat[:3, :3] = R_inv
    viewmat[:3, 3:4] = T_inv
    # calculate the FOV of the camera given fx and fy, width and height
    cx = camera.cx.item()
    cy = camera.cy.item()
    W, H = int(camera.width.item()), int(camera.height.item())
    self.last_size = (H, W)

    if crop_ids is not None:
        opacities_crop = self.opacities[crop_ids]
        means_crop = self.means[crop_ids]
        features_dc_crop = self.features_dc[crop_ids]
        features_rest_crop = self.features_rest[crop_ids]
        scales_crop = self.scales[crop_ids]
        quats_crop = self.quats[crop_ids]
    else:
        opacities_crop = self.opacities
        means_crop = self.means
        features_dc_crop = self.features_dc
        features_rest_crop = self.features_rest
        scales_crop = self.scales
        quats_crop = self.quats

    un_points = self.get_uncertainty(means_crop).view(-1)

    # Normalize the uncertainty values between 0 and 1
    un_points_min = un_points.min()
    un_points_max = un_points.max()

    un_points_cp = (un_points - un_points_min) / (un_points_max - un_points_min)

    # Filter out Gaussians with uncertainty greater than the threshold
    valid_indices = un_points_cp <= self.filter_thresh
    opacities_crop = opacities_crop[valid_indices]
    means_crop = means_crop[valid_indices]
    features_dc_crop = features_dc_crop[valid_indices]
    features_rest_crop = features_rest_crop[valid_indices]
    scales_crop = scales_crop[valid_indices]
    quats_crop = quats_crop[valid_indices]
    un_points = un_points[valid_indices]

    colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
    BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
    self.xys, depths, self.radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
        means_crop,
        torch.exp(scales_crop),
        1,
        quats_crop / quats_crop.norm(dim=-1, keepdim=True),
        viewmat.squeeze()[:3, :],
        camera.fx.item(),
        camera.fy.item(),
        cx,
        cy,
        H,
        W,
        BLOCK_WIDTH,
    )  # type: ignore

    # rescale the camera back to original dimensions before returning
    camera.rescale_output_resolution(camera_downscale)

    if (self.radii).sum() == 0:
        return self.get_empty_outputs(W, H, background)

    if self.config.sh_degree > 0:
        viewdirs = means_crop.detach() - optimized_camera_to_world.detach()[:3, 3]  # (N, 3)
        n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        rgbs = spherical_harmonics(n, viewdirs, colors_crop)  # input unnormalized viewdirs
        rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
    else:
        rgbs = torch.sigmoid(colors_crop[:, 0, :])

    assert (num_tiles_hit > 0).any()  # type: ignore

    # apply the compensation of screen space blurring to gaussians
    if self.config.rasterize_mode == "antialiased":
        opacities = torch.sigmoid(opacities_crop) * comp[:, None]
    elif self.config.rasterize_mode == "classic":
        opacities = torch.sigmoid(opacities_crop)
    else:
        raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

    rgb, alpha = rasterize_gaussians(  # type: ignore
        self.xys,
        depths,
        self.radii,
        conics,
        num_tiles_hit,  # type: ignore
        rgbs,
        opacities,
        H,
        W,
        BLOCK_WIDTH,
        background=background,
        return_alpha=True,
    )  # type: ignore
    alpha = alpha[..., None]
    rgb = torch.clamp(rgb, max=1.0)  # type: ignore
    depth_im = None
    uncertainty_im = None

    # #normalize into acceptable range for rendering
    # uncertainty = torch.clip(uncertainty, min_uncertainty, max_uncertainty)
    # uncertainty = (uncertainty-min_uncertainty)/(max_uncertainty-min_uncertainty)

    # breakpoint()

    if self.config.output_depth_during_training or not self.training:
        depth_im = rasterize_gaussians(  # type: ignore
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,  # type: ignore
            depths[:, None].repeat(1, 3),
            opacities,
            H,
            W,
            BLOCK_WIDTH,
            background=torch.zeros(3, device=self.device),
        )[..., 0:1]  # type: ignore
        uncertainty_im = rasterize_gaussians(  # type: ignore
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,  # type: ignore
            un_points[:, None].repeat(1, 3),
            opacities,
            H,
            W,
            BLOCK_WIDTH,
            background=torch.zeros(3, device=self.device),
        )[..., 0:1]  # type: ignore
        depth_im = torch.where(alpha > 0, depth_im / alpha, depth_im.detach().max())
        # breakpoint()
        nan_tensor = torch.full_like(uncertainty_im, -4)
        uncertainty_im = torch.where(alpha > 0, uncertainty_im / alpha, nan_tensor)
        uncertainty_im = torch.clip(uncertainty_im, min_uncertainty, max_uncertainty)
        uncertainty_im = (uncertainty_im - min_uncertainty) / (max_uncertainty - min_uncertainty)

    return {"rgb": rgb, "depth": depth_im, "accumulation": alpha, "uncertainty": uncertainty_im,
            "background": background}  # type: ignore
