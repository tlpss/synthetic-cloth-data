#!/bin/bash
#python generate_data.py --dataset-size 10  --hydra_config tshirts --hydra_args +experiment=tshirts-single-layer-random-material
python generate_data.py --dataset-size 5000  --hydra_config tshirts --hydra_args +experiment=tshirts-single-layer-flat-random-material
python generate_data.py --dataset-size 5000  --hydra_config tshirts --hydra_args +experiment=tshirts-cloth3d-random-material
