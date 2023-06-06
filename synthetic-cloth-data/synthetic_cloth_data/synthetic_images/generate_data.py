import pathlib
import subprocess

import click
from tqdm import tqdm

script = pathlib.Path(__file__).parent / "scene_builder" / "create_cloth_scene.py"


@click.command()
@click.option("--dataset-size", default=10, help="Number of samples to generate")
@click.option("--start_id", default=0, help="Start id for the dataset")
@click.option("--hydra_config", type=str, default="dev")
@click.option(
    "--hydra", multiple=True, default=[]
)  # cannot have list directly in click, so use multiple flag and combine later
def generate_cloth_data(dataset_size: int, start_id: int, hydra_config: str, hydra: list):
    print("generating cloth data")
    log_file = pathlib.Path("data-gen.log").open("w")
    for seed in tqdm(range(start_id, start_id + dataset_size)):
        command = f"blender -b -P {script} -- --hydra_config {hydra_config} --hydra id={seed}"
        for hydra_arg in hydra:
            command += f" {hydra_arg}"
        subprocess.run([command], shell=True, stdout=log_file, stderr=log_file)


if __name__ == "__main__":
    generate_cloth_data()
