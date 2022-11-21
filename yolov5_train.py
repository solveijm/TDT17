import os

os.system("python ./yolov5/train.py --img 640 --batch 16 --epochs 300 --data coco128.yaml --weights yolov5s.pt --device 0")