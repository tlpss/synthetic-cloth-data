import json
import multiprocessing as mp
import pathlib
import subprocess

import click
import numpy as np
import tqdm
from synthetic_cloth_data import DATA_DIR
from synthetic_cloth_data.utils import get_metadata_dict_for_dataset

deformation_script = pathlib.Path(__file__).parent / "blender_deform_mesh.py"


def func(id_range, mesh_dir, output_dir):
    for id in tqdm.tqdm(id_range):
        command = f"blender -b -P {deformation_script} -- --id {id} --mesh_dir_relative_path {mesh_dir} --output_dir {output_dir}"
        print(command)
        subprocess.run([command], shell=True)


@click.command()
@click.option("--mesh-dir", type=str, default="flat_meshes/TOWEL/dev")
@click.option("--num-samples", type=int, default=20)
@click.option("--output-dir", type=str, default="deformed_meshes/TOWEL/dev")
@click.option("--n-workers", type=int, default=1)
def generate_dataset(mesh_dir: str, num_samples: int, output_dir: str, n_workers: int):

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

    if n_workers > 1:

        pool = mp.Pool(n_workers)

        # split range in n_workers
        id_range = range(num_samples)
        id_range_split = np.array_split(id_range, n_workers)
        for id in id_range_split:
            pool.apply_async(func, args=(id, mesh_dir, output_dir))
        pool.close()
        pool.join()
    else:
        func(range(num_samples), mesh_dir, output_dir)


if __name__ == "__main__":
    generate_dataset()
