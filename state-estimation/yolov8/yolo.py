import yaml
from ultralytics import YOLO
from ultralytics.data import YOLODataset

path = "data/yolo/real-towels.yaml"
yaml_dict = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
dataset = YOLODataset(data=yaml_dict)

model = YOLO(path)
model.train()
