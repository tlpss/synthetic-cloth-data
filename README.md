# synthetic-cloth-data

This repo contains a Blender-based pipeline for procedural data generation of clothes to learn perception models for robotic cloth manipulation.

It contains code to
- generate single-layer cloth meshes & their keypoint vertex IDs using procedural geometric templates
- deform cloth meshes using Nvidia Flex or the blender cloth simulator
-create scenes in blender and to render images and their corresponding COCO annotations

The data generation pipeline can be easily extended to other cloth types, other deformation configs (e.g. hanging from a robot gripper) and/or other data modalities (e.g. depth images...) with the corresponding annotations.

Furthermore, the repo also contains code to reproduce all experiments from the [accompanying paper]() and a number of pretrained models for keypoint detection that were used  to evaluate the synthetic data generation pipeline. See [here]() for how to use these models.


## Installation
Take the following steps to set up this repo locally:

- git clone submodules recurse: `git clone <repo> --recurse-submodules`
- create conda env: `conda env create -f environment.yaml`
- install & setup blender in airo-blender: from the `airo-blender/blender` folder, run `source ../bash_scripts/setup_blender.sh << path-to-your-conda-env`
- add blender to your path: `source add_blender_to_path.sh`
- download the blender assets that are used during the synthetic data generation, see [here](synthetic-cloth-data/synthetic_cloth_data/synthetic_images/assets/readme.md)
- (if you need it) install pyflex by following the instructions [here](pyflex/Readme.md)

## Generating Synthetic Data
The data generation pipeline has three steps:

1. obtain flat meshes
2. deform the meshes
3. render images and create the corresponding annotations

### Flat meshes

### Deformed meshes
pyflex, hydra configs

### Synthetic Images
blender, hydra configs, visibility checking.


some pointers:
- generate flat meshes : `blender -b -P meshes/generate_flat_meshes.py -- --cloth_type TOWEL --num_samples 4 --dataset_tag dev`
- generate deformed meshes with pyflex: `python meshes/pyflex_deformations/pyflex_deform_mesh.py`
- visualize meshes in blender: `blender -P visualize_meshes.py`
- generate synthetic images: `synthetic_images/generate_data.py`. This uses hydra to configure various aspects of the pipeline.
- create coco json file: `synthetic_images/combine_samples_to_coco_dataset.py`


# Reproducing the Paper
This section contains all the commands to generate the datasets and train the keyppoints to reproduce all experiments and results described in the paper.
<details>
   <summary>Click here to expand</summary>

## Data Generation

### flat meshes
single-layer tshirts:

cloth3d tshirts:

single-layer Towels: `blender -b -P generate_flat_meshes.py -- --cloth_type TOWEL --num_samples 1000 --dataset_tag 00-final`

single-layer Shorts:

### Deformed meshes
single-layer Tshirts:

cloth3d Tshirts:

single-layer Towels:

single-layer Shorts:

### Synthetic Images
All these commands should be run from the `synthetic_cloth_data/synthetic_images` folder.

Tshirts-main: `python generate_data.py --dataset-size 11111 --start_id 0  --hydra_config tshirts --hydra_args +experiment=tshirts-main`

Towels-main: `python generate_data.py --dataset-size 11111 --start_id 0  --hydra_config towels --hydra_args +experiment=towels-main`

Shorts-main: `python generate_data.py --dataset-size 11111 --start_id 0  --hydra_config shorts --hydra_args +experiment=shorts-main`


**Mesh comparisons**

*cloth3d meshes*

generate the deformed meshes:

generate the dataset:

*undeformed meshes*

generate the dataset:


**Material comparisons**

uniform materials:

tailored materials:


## Training the Keypoint Detectors
all code for training the keypoint detectors that are used in the paper is located in the `state-estimation/state_estimation/keypoint_detection` folder. All commands are run from this folder. A [Keypoint Detection framework](https://github.com/tlpss/keypoint-detection) is used to train the keypoint detectors.


**Installation**
- `pip install -r requirements.txt`
- log in to wandb: `wandb login`

**Gathering the data**

*aRTF dataset*

Download all splits from [TODO]() and place in `state-estimation/state_estimation/data/artf` folder.

*Synthetic datasets*

Either generate the appropriate synthetic datasets as described above and symlink the `synthetic-cloth-data/data/datasets` folder to `state-estimation/state_estimation/data/synthetic/`

or download the datasets from [TODO]() and place in `state-estimation/state_estimation/data/synthetic` folder.

**Main results**

real baseline: `python real_baselines.py`

sim_to_real: `python synthetic_main.py`

finetune ^ on real: `python synthetic-finetune-main.py`.
 Don't forget to update the wandb artifacts if you trained a new sim-to-real model.

 sim-to-sim: `python synthetic-sim-validation-main.py`

**Additional experiments**

mesh comparisons: `python tshirts_meshes.py`

material comparisons: `python tshirts_materials.py`


</details>
