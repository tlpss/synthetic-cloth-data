import pathlib
import subprocess

import click
from tqdm import tqdm

script = pathlib.Path(__file__).parent / "create_cloth_scene.py"


@click.command()
@click.option("--dataset-size", default=10, help="Number of samples to generate")
def generate_cloth_data(dataset_size: int):
    print("generating cloth data")
    for seed in tqdm(range(dataset_size)):
        command = f"blender -b -P {script} -- --id {seed}"
        subprocess.run([command], shell=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    generate_cloth_data()
