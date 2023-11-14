This package contains the configurations and commands to train the keypoint detection results that are presented in the paper.

## setup

- move into the `state-estimation` folder
- install this package (only serves to make imports easier): `pip install -e .`
- symlink the aRTF clothes dataset into `state_estimation/data/artf_data`
- symlink the `synthetic-cloth-data/data/datasets` into `state_estimation/data/synthetic-data`

### Keypoint detector
- pip install the keypoint detector using `pip install -e state_estimation/keypoint_detection/requirements.txt`
- log in to your wandb account



## Main results

### real baselines
`python real_baselines.py`
### sim2real
### sim2sim
### sim2real + finetuning

## dataset size experiments

## pipeline evaluation
### materials
### meshes


