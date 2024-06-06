import time
from pathlib import Path

import pkg_resources
import tyro
from dataclasses import dataclass
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
