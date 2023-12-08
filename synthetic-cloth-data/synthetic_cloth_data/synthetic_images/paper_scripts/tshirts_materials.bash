#!/bin/bash

python generate_data.py --dataset-size 5000  --hydra_config tshirts --hydra_args +experiment=tshirts-single-layer-full-material
python generate_data.py --dataset-size 5000  --hydra_config tshirts --hydra_args +experiment=tshirts-single-layer-hsv-material
python generate_data.py --dataset-size 5000  --hydra_config tshirts --hydra_args +experiment=tshirts-single-layer-random-material
