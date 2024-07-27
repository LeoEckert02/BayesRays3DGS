import time
from pathlib import Path

import pkg_resources
import torch
import tyro
from dataclasses import dataclass
import numpy as np

from gsplat import spherical_harmonics
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.model_components import renderers
from nerfstudio.utils.eval_utils import eval_setup

from bayessplatting.utils.utils import find_grid_indices, normalize_point_coords
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians


@dataclass
class ComputeUncertainty:
    """Load a checkpoint, compute uncertainty, and save it to a npy file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("unc.npy")
    # Uncertainty level of detail (log2 of it)
    lod: int = 8
    # number of iterations on the trainset
    iters: int = 1000

    def find_uncertainty(self, points, deform_points, rgb):
        inds, coeffs = find_grid_indices(points, self.lod, self.device)
        # breakpoint()
        # because deformation params are detached for each point on each ray from the grid, summation does not affect derivative
        colors = torch.sum(rgb, dim=0)
        colors[0].backward(retain_graph=True)
        r = deform_points.grad.clone().detach().view(-1, 3)

        deform_points.grad.zero_()
        colors[1].backward(retain_graph=True)
        g = deform_points.grad.clone().detach().view(-1, 3)

        deform_points.grad.zero_()
        colors[2].backward()
        b = deform_points.grad.clone().detach().view(-1, 3)

        deform_points.grad.zero_()
        dmy = torch.arange(inds.shape[1], device=self.device)
        first = True
        for corner in range(8):
            if first:
                all_ind = torch.cat((dmy.unsqueeze(-1), inds[corner].unsqueeze(-1)), dim=-1)
                all_r = coeffs[corner].unsqueeze(-1) * r
                all_g = coeffs[corner].unsqueeze(-1) * g
                all_b = coeffs[corner].unsqueeze(-1) * b
                first = False
            else:
                all_ind = torch.cat((all_ind, torch.cat((dmy.unsqueeze(-1), inds[corner].unsqueeze(-1)), dim=-1)),dim=0)
                all_r = torch.cat((all_r, coeffs[corner].unsqueeze(-1) * r), dim=0)
                all_g = torch.cat((all_g, coeffs[corner].unsqueeze(-1) * g), dim=0)
                all_b = torch.cat((all_b, coeffs[corner].unsqueeze(-1) * b), dim=0)

        keys_all, inds_all = torch.unique(all_ind, dim=0, return_inverse=True)
        grad_r_1 = torch.bincount(inds_all, weights=all_r[..., 0])  # for first element of deformation field
        grad_g_1 = torch.bincount(inds_all, weights=all_g[..., 0])
        grad_b_1 = torch.bincount(inds_all, weights=all_b[..., 0])
        grad_r_2 = torch.bincount(inds_all, weights=all_r[..., 1])  # for second element of deformation field
        grad_g_2 = torch.bincount(inds_all, weights=all_g[..., 1])
        grad_b_2 = torch.bincount(inds_all, weights=all_b[..., 1])
        grad_r_3 = torch.bincount(inds_all, weights=all_r[..., 2])  # for third element of deformation field
        grad_g_3 = torch.bincount(inds_all, weights=all_g[..., 2])
        grad_b_3 = torch.bincount(inds_all, weights=all_b[..., 2])
        grad_1 = grad_r_1 ** 2 + grad_g_1 ** 2 + grad_b_1 ** 2
        grad_2 = grad_r_2 ** 2 + grad_g_2 ** 2 + grad_b_2 ** 2
        grad_3 = grad_r_3 ** 2 + grad_g_3 ** 2 + grad_b_3 ** 2  # will consider the trace of each submatrix for each deformation
        # vector as indicator of hessian wrt the whole vector

        grads_all = torch.cat((keys_all[:, 1].unsqueeze(-1), (grad_1 + grad_2 + grad_3).unsqueeze(-1)), dim=-1)

        hessian = torch.zeros(((2 ** self.lod) + 1) ** 3).to(self.device)
        hessian = hessian.put((grads_all[:, 0]).long(), grads_all[:, 1], True)


        return hessian

    def main(self) -> None:
        """Main function."""

        if pkg_resources.get_distribution("nerfstudio").version >= "1.1.0":
            config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)
        else:
            config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        self.device = pipeline.device
        self.hessian = torch.zeros(((2 ** self.lod) + 1) ** 3).to(self.device)
        self.deform_field = HashEncoding(num_levels=1,
                                         min_res=2 ** self.lod,
                                         max_res=2 ** self.lod,
                                         log2_hashmap_size=self.lod * 3 + 1,
                                         # simple regular grid (hash table size > grid size)
                                         features_per_level=3,
                                         hash_init_scale=0.,
                                         implementation="torch",
                                         interpolation="Linear")
        self.deform_field.to(self.device)
        self.deform_field.scalings = torch.tensor([2 ** self.lod]).to(self.device)

        # breakpoint()

        pipeline.eval()
        len_train = max(pipeline.datamanager.train_dataset.__len__(), self.iters)
        for step in range(len_train):
            print("step", step)
            camera, _ = pipeline.datamanager.next_train(step)
            # breakpoint()
            outputs, points, offsets = self.get_outputs(camera, pipeline.model)
            # breakpoint()
            hessian = self.find_uncertainty(points, offsets, outputs['rgb'].view(-1, 3))
            self.hessian += hessian.clone().detach()

        end_time = time.time()
        print("Done")
        with open(str(self.output_path), 'wb') as f:
            np.save(f, self.hessian.cpu().numpy())
        execution_time = end_time - start_time
        # breakpoint()
        print(f"Execution time: {execution_time:.6f} seconds")

    def get_outputs(self, camera: Cameras, model):
        """Reimplementation of the get_ouptut mehtosd to add offsets and points"""
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"

        # get the background color
        if model.training:
            optimized_camera_to_world = model.camera_optimizer.apply_to_camera(camera)[0, ...]

            if model.config.background_color == "random":
                background = torch.rand(3, device=self.device)
            elif model.config.background_color == "white":
                background = torch.ones(3, device=self.device)
            elif model.config.background_color == "black":
                background = torch.zeros(3, device=self.device)
            else:
                background = model.background_color.to(self.device)
        else:
            optimized_camera_to_world = camera.camera_to_worlds[0, ...]

            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device)
            else:
                background = model.background_color.to(self.device)

        if model.crop_box is not None and not model.training:
            crop_ids = model.crop_box.within(model.means).squeeze()
            if crop_ids.sum() == 0:
                return model.get_empty_outputs(int(camera.width.item()), int(camera.height.item()), background)
        else:
            crop_ids = None
        camera_downscale = self.get_downscale_factor(model)
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
            opacities_crop = model.opacities[crop_ids]
            means_crop = model.means[crop_ids]
            features_dc_crop = model.features_dc[crop_ids]
            features_rest_crop = model.features_rest[crop_ids]
            scales_crop = model.scales[crop_ids]
            quats_crop = model.quats[crop_ids]
        else:
            opacities_crop = model.opacities
            means_crop = model.means
            features_dc_crop = model.features_dc
            features_rest_crop = model.features_rest
            scales_crop = model.scales
            quats_crop = model.quats

        # breakpoint()
        # get the offsets from the deform field
        normalized_points = normalize_point_coords(means_crop)
        offsets = self.deform_field(normalized_points).clone().detach()
        offsets.requires_grad = True
        # breakpoint()

        means_crop = means_crop + offsets

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        model.xys, depths, model.radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
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

        if (model.radii).sum() == 0:
            return model.get_empty_outputs(W, H, background)

        if model.config.sh_degree > 0:
            viewdirs = means_crop.detach() - optimized_camera_to_world.detach()[:3, 3]  # (N, 3)
            n = min(model.step // model.config.sh_degree_interval, model.config.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors_crop)  # input unnormalized viewdirs
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
        else:
            rgbs = torch.sigmoid(colors_crop[:, 0, :])

        assert (num_tiles_hit > 0).any()  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if model.config.rasterize_mode == "antialiased":
            opacities = torch.sigmoid(opacities_crop) * comp[:, None]
        elif model.config.rasterize_mode == "classic":
            opacities = torch.sigmoid(opacities_crop)
        else:
            raise ValueError("Unknown rasterize_mode: %s", model.config.rasterize_mode)

        rgb, alpha = rasterize_gaussians(  # type: ignore
            model.xys,
            depths,
            model.radii,
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
        if model.config.output_depth_during_training or not model.training:
            depth_im = rasterize_gaussians(  # type: ignore
                model.xys,
                depths,
                model.radii,
                conics,
                num_tiles_hit,  # type: ignore
                depths[:, None].repeat(1, 3),
                opacities,
                H,
                W,
                BLOCK_WIDTH,
                background=torch.zeros(3, device=self.device),
            )[..., 0:1]  # type: ignore
            depth_im = torch.where(alpha > 0, depth_im / alpha, depth_im.detach().max())

        outputs = {
            "rgb": rgb,
            "depth": depth_im,
            "accumulation": alpha,
            "background": background
        }

        # means_crop are xyz points of gaussian splat
        return outputs, means_crop, offsets

    def get_downscale_factor(self, model):
        if model.training:
            return 2 ** max(
                (model.config.num_downscales - model.step // model.config.resolution_schedule),
                0,
            )
        else:
            return 1


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputeUncertainty).main()


if __name__ == '__main__':
    entrypoint()
