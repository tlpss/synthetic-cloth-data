import pathlib
import subprocess

import click
from synthetic_cloth_data.synthetic_images.combine_samples_to_coco_dataset import (
    create_coco_dataset_from_intermediates,
)
from tqdm import tqdm

script = pathlib.Path(__file__).parent / "scene_builder" / "create_cloth_scene.py"
from synthetic_cloth_data import DATA_DIR


@click.command()
@click.option("--dataset-size", default=10, help="Number of samples to generate")
@click.option("--start_id", default=0, help="Start id for the dataset")
@click.option("--hydra_config", type=str, default="dev")
@click.option(
    "--hydra_args", multiple=True, default=[]
)  # cannot have list directly in click, so use multiple flag and combine later
def generate_cloth_data(dataset_size: int, start_id: int, hydra_config: str, hydra_args: list):
    import hydra
    from omegaconf import OmegaConf

    # initialize hydra
    hydra.initialize(config_path="configs", job_name="create_cloth_scene")
    cfg = hydra.compose(config_name=hydra_config, overrides=hydra_args)
    print(f"hydra config: \n{OmegaConf.to_yaml(cfg)}")

    dataset_dir = cfg["relative_dataset_path"]
    assert "synthetic_images" in dataset_dir, "relative dataset dir must be in synthetic_images data folder."

    print("generating cloth data")
    log_file = pathlib.Path("data-gen.log").open("w")
    for seed in tqdm(range(start_id, start_id + dataset_size)):
        command = f"blender -b -P {script} -- --hydra_config {hydra_config} --hydra id={seed}"
        for hydra_arg in hydra_args:
            command += f" {hydra_arg}"
        subprocess.run([command], shell=True, stdout=log_file, stderr=log_file)

    # combine samples to coco dataset
    target_dir = str(dataset_dir).replace("synthetic_images", "datasets")
    create_coco_dataset_from_intermediates(target_dir, str(dataset_dir))

    # split dataset into train and val
    # this does not really belong here, but it makes life easier.
    from airo_dataset_tools.coco_tools.split_dataset import split_and_save_coco_dataset

    split_and_save_coco_dataset(
        str(DATA_DIR / target_dir / "annotations.json"), [0.9, 0.1], shuffle_before_splitting=True
    )


if __name__ == "__main__":
    generate_cloth_data()
