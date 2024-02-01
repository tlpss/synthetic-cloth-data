# State estimation

## setup

- move into the `state-estimation` folder
- install this package (only serves to make imports easier): `pip install -e .`

*aRTF dataset*

Download all splits from [the aRTF Clothes repo](https://github.com/tlpss/aRTF-Clothes-dataset/tree/main?tab=readme-ov-file#using-this-dataset) and put them in the `state-estimation/state_estimation/data/artf_data` folder.

*Synthetic datasets*

Either generate the appropriate synthetic datasets as described above and symlink the `synthetic-cloth-data/data/datasets` folder to `state-estimation/state_estimation/data/synthetic-data/`

or download the datasets from [TODO]() and place in `state-estimation/state_estimation/data/synthetic-data` folder.

## Keypoint detector

### Installation
- pip install the keypoint detector using `pip install -e state_estimation/keypoint_detection/requirements.txt`
- log in to your wandb account


### Pretrained checkpoints

 The checkpoints are listed in the `final_checkpoints.py` file.

To use them:
```python

from keypoint_detection.utils.load_checkpoints import get_model_from_wandb_checkpoint

get_model_from_wandb_checkpoint(<final-checkpoint-string>)

```

An example of how to detect keypoints in an image using these model checkpoints can be found at `state_estimation/keypoint_detection/visualizations/visualisation.py`

### Experiments
scripts for reproducing the main results can be found below.
All checkpoints as well as the wandb training runs are also available.

real baselines: `python real_baselines.py`

sim2real:`python synthetic_main.py`

sim2sim: `python synthetic-sim-validation-main.py`

training sim checkpoints on real data: `python synthetic-finetune-main.py`

**additional experiments**

meshes: `python tshirts_materials.py`

materials: `python tshirts_meshes.py`


### visualizations

to generate the tables from the paper:
`python results/generate_main_results_table.py`

to generate the figures, see the `visualizations/` dir.
