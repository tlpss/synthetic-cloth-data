# State estmation

## setup

- move into the `state-estimation` folder
- install this package (only serves to make imports easier): `pip install -e .`
- symlink the aRTF clothes dataset into `state_estimation/data/artf_data`
- symlink the `synthetic-cloth-data/data/datasets` into `state_estimation/data/synthetic-data`


## Keypoint detector

### Installation
- pip install the keypoint detector using `pip install -e state_estimation/keypoint_detection/requirements.txt`
- log in to your wandb account


### Main results

scripts for reproducing the main results can be found below.
All checkpoints as well as the wandb training runs are also available. The checkpoints are listed in the `final_checkpoints.py` file.
See the visualization scripts below for an example of how to use them.

### real baselines
`python real_baselines.py`
### sim2real
`python synthetic_main.py`
### sim2sim
`python synthetic-sim-validation-main.py`
### training sim checkpoints on real data
`python synthetic-finetune-main.py`

## pipeline evaluation


### meshes
`python tshirts_materials.py`
### materials
`python tshirts_meshes.py`


## visualizations

to generate the tables from the paper:
`python results/generate_main_results_table.py`

to generate the figures, see the `visulaziations/` dir.
