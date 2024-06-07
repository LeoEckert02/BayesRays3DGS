import time
from pathlib import Path

import pkg_resources
import torch
import tyro
from dataclasses import dataclass

from nerfstudio.field_components.encodings import HashEncoding
from nerfstudio.utils.eval_utils import eval_setup



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

    def main(self) -> None:
        """Main function."""

        if pkg_resources.get_distribution("nerfstudio").version >= "1.1.0":
            config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)
        else:
            config, pipeline, checkpoint_path = eval_setup(self.load_config)

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

        pipeline.eval()
        len_train = max(pipeline.datamanager.train_dataset.__len__(), self.iters)
        for step in range(len_train):
            print("step", step)
            ray_bundle, batch = pipeline.datamanager.next_train(step)
            output_fn = self.get_output_fn(pipeline.model)
            outputs, points, offsets, = output_fn(ray_bundle, pipeline.model)
            hessian = self.find_uncertainty(points_fine, offsets_fine, outputs['rgb_fine'],
                                            pipeline.model.field.spatial_distortion)
            self.hessian += hessian.clone().detach()
            hessian = self.find_uncertainty(points_coarse, offsets_coarse, outputs['rgb_coarse'],
                                            pipeline.model.field.spatial_distortion)
            self.hessian += hessian.clone().detach()

        end_time = time.time()
        print("Done")
        # with open(str(self.output_path), 'wb') as f:
        #     np.save(f, self.hessian.cpu().numpy())
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.6f} seconds")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputeUncertainty).main()


if __name__ == '__main__':
    entrypoint()
