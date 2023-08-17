import json
import pathlib

import click
import tqdm
from synthetic_cloth_data import DATA_DIR
from synthetic_cloth_data.meshes.pyflex_deformations.pyflex_deform_mesh import generate_deformed_mesh
from synthetic_cloth_data.utils import get_metadata_dict_for_dataset


@click.command()
@click.option("--mesh-dir", type=str, default="flat_meshes/TOWEL/dev")
@click.option("--num-samples", type=int, default=20)
@click.option("--output-dir", type=str, default="deformed_meshes/TOWEL/pyflex/dev")
@click.option("--start_id", type=int, default=0)
def generate_dataset(mesh_dir: str, num_samples: int, output_dir: str, start_id: int):

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

    for id in tqdm.tqdm(range(start_id,start_id+num_samples)):
        generate_deformed_mesh(mesh_dir, output_dir, id, debug=False)


if __name__ == "__main__":
    generate_dataset()
