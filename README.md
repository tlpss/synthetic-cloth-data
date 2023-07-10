# synthetic-cloth-data
This repo facilitates procedural data generation to learn perception modules for cloth manipulation.

It contains
- geometric templates for cloth items + bevels & bezier curves to generate flat meshes & their keypoint vertex IDs
- code for deforming these meshes using PyFlex or the blender cloth simulator
- code for cloth materials in blender
- code for rendering images and creating COCO datasets.

## Installation
- git clone submodules recurse
- create conda env: `conda env create -f environment.yaml`
- install & setup blender in airo-blender: from the `airo-blender/blender` folder, run `source ../bash_scripts/setup_blender.sh << path-to-your-conda-env`
- add blender to your path: `source add_blender_to_path.sh`
- (if you need it) install pyflex by following the instructions [here](pyflex/Readme.md)

## Generating Data

some pointers:
- generate flat meshes : `blender -b -P meshes/generate_flat_meshes.py -- --cloth_type TOWEL --num_samples 4 --dataset_tag dev`
- generate deformed meshes with pyflex: `python meshes/pyflex_deformations/pyflex_deform_mesh.py`
- visualize meshes in blender: `blender -P visualize_meshes.py`
- generate synthetic images: `synthetic_images/generate_data.py`. This uses hydra to configure various aspects of the pipeline.
- create coco json file: `synthetic_images/combine_samples_to_coco_dataset.py`


