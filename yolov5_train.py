import os

os.system("python ./yolov5/train.py --img 640 --batch 32 --epochs 300 --data dataset.yaml --weights yolov5x.pt --device 0")