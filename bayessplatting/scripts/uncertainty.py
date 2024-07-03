import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pkg_resources
import torch
import tyro
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.misc import torch_compile

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

from bayessplatting.utils.utils import find_grid_indices, normalize_point_coords


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
        inds, coeffs = find_grid_indices(points, self.aabb, self.lod, self.device)
        # breakpoint()
        # because deformation params are detached for each point on each ray from the grid, summation does not affect derivative
        colors = torch.sum(rgb, dim=0)
        colors[0].backward(retain_graph=True)
        r = deform_points.grad.clone().detach().view(-1, 3)
        print("Offset of the points (deform_points):", deform_points)

        deform_points.grad.zero_()
        colors[1].backward(retain_graph=True)
        g = deform_points.grad.clone().detach().view(-1, 3)
        print("Gradient after first backward pass (g):", g)

        deform_points.grad.zero_()
        colors[2].backward()
        b = deform_points.grad.clone().detach().view(-1, 3)
        print("Gradient after third backward pass (b):", b)

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
                all_ind = torch.cat((all_ind, torch.cat((dmy.unsqueeze(-1), inds[corner].unsqueeze(-1)), dim=-1)),
                                    dim=0)
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
        print("Gradients for each deformation component combined (grads_all):", grads_all)

        hessian = torch.zeros(((2 ** self.lod) + 1) ** 3).to(self.device)
        hessian = hessian.put((grads_all[:, 0]).long(), grads_all[:, 1], True)
        print("Hessian:", hessian)

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
        self.aabb = pipeline.model.scene_box.aabb.to(self.device)
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
        breakpoint()
        print(f"Execution time: {execution_time:.6f} seconds")

    def get_outputs(self, camera: Cameras, model):
        """Reimplementation of the get_outputs method to add offsets and points"""
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if model.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = model.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # cropping
        if model.crop_box is not None and not model.training:
            crop_ids = model.crop_box.within(model.means).squeeze()
            if crop_ids.sum() == 0:
                return model.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), model.background_color
                )
        else:
            crop_ids = None

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
        normalized_points, _ = normalize_point_coords(means_crop, self.aabb)
        offsets = self.deform_field(normalized_points).clone().detach()
        offsets.requires_grad = True
        # breakpoint()

        means_crop = means_crop + offsets

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        camera_scale_fac = model._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = self.get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        model.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if model.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", model.config.rasterize_mode)

        if model.config.output_depth_during_training or not model.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if model.config.sh_degree > 0:
            sh_degree_to_use = min(model.step // model.config.sh_degree_interval, model.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop)
            sh_degree_to_use = None

        render, alpha, info = rasterization(
            means=means_crop,
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=model.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )
        if model.training and info["means2d"].requires_grad:
            info["means2d"].retain_grad()
        model.xys = info["means2d"]  # [1, N, 2]
        model.radii = info["radii"][0]  # [N]
        alpha = alpha[:, ...]

        background = model.background_color.to(self.device)
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        if background.shape[0] == 3 and not model.training:
            background = background.expand(H, W, 3)

        outputs = {
            "rgb": rgb,
            "depth": depth_im,
            "accumulation": alpha,
            "background": background
        }

        # means_crop are xyz points of gaussian splat
        return outputs, means_crop, offsets

    @torch_compile()
    def get_viewmat(optimized_camera_to_world):
        """
        function that converts c2w to gsplat world2camera matrix, using compile for some speed
        """
        R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
        T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
        # flip the z and y axes to align with gsplat conventions
        R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.transpose(1, 2)
        T_inv = -torch.bmm(R_inv, T)
        viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
        viewmat[:, 3, 3] = 1.0  # homogenous
        viewmat[:, :3, :3] = R_inv
        viewmat[:, :3, 3:4] = T_inv
        return viewmat

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
