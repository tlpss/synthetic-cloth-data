import json
import pathlib
import subprocess

import click
import tqdm
from synthetic_cloth_data import DATA_DIR
from synthetic_cloth_data.utils import get_metadata_dict_for_dataset

deformation_script = pathlib.Path(__file__).parent / "blender_deform_mesh.py"


@click.command()
@click.option("--mesh-dir", type=str, default="flat_meshes/TOWEL/dev")
@click.option("--num-samples", type=int, default=20)
@click.option("--output-dir", type=str, default="deformed_meshes/TOWEL/dev")
def generate_dataset(mesh_dir: str, num_samples: int, output_dir: str):
    # write metadata
    data = {
        "num_samples": num_samples,
        "flat_mesh_dir": mesh_dir,
    }
    data.update(get_metadata_dict_for_dataset())
    output_dir = DATA_DIR / pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"
    json.dump(data, open(metadata_path, "w"))
    print(f"Metadata written to {metadata_path}")

    for id in tqdm.trange(num_samples):
        command = f"blender -b -P {deformation_script} -- --id {id} --mesh_dir_relative_path {mesh_dir} --output_dir {output_dir}"
        print(command)
        subprocess.run([command], shell=True)  # stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    generate_dataset()
