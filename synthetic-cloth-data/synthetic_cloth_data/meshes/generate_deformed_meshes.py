import pathlib
import subprocess

import tqdm
from synthetic_cloth_data.meshes.cloth_meshes import CLOTH_TYPES

towel_script = pathlib.Path(__file__).parent / "deformed_towel.py"


def generate_dataset(cloth_type: CLOTH_TYPES, num_samples: int, output_dir: str):
    output_dir = pathlib.Path(output_dir) / cloth_type.name
    output_dir.mkdir(parents=True, exist_ok=True)
    # TODO: pass output DIR and other params in any fricking way that allows blender to be called from a subprocess
    # TODO: deal with multiple cloth types
    for id in tqdm.trange(num_samples):
        command = f"blender -b -P {towel_script} -- --id {id}"
        subprocess.run([command], shell=True)  # stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":

    from synthetic_cloth_data import DATA_DIR

    output_dir = DATA_DIR / "deformed_meshes"
    num_samples = 20
    cloth_type = CLOTH_TYPES.TOWEL
    generate_dataset(cloth_type, num_samples, output_dir)
