# synthetic-cloth-data
This repo facilitates procedural data generation for keypoint detection on clothes.

It contains
- geometric templates for cloth items + bevels & bezier curves to generate flat meshes & their keypoint vertex IDs
- code for deforming these meshes using the blender cloth simulator
- code for cloth materials in blender
- code for creating COCO datasets of the cloth items.

## Installation
- git clone submodules recurse
- create conda env: `conda env create -f environment.yaml`
- install & setup blender in airo-blender: from the `airo-blender/blender` folder, run `source ../bash_scripts/setup_blender.sh << path-to-your-conda-env`

## Generating Data
- generate flat meshes : `blender -b -P meshes/generate_flat_meshes.py -- --cloth_type TOWEL --num_samples 4 --dataset_tag dev`
- generate deformed meshes
- generate synthetic images
- create coco json file


## Development

### Running formatting, linting and testing
The makefile contains commands to make this convenient. Run using `make <command>`.