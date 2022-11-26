# TDT17 - Mini project
Crack detection on roads using the RDD2022-dataset.

### Statistics of the dataset and preprocessing
See the file crackDetection.ipynb

### Train the model
The following command was used to train the model
'''
python ./yolov5/train.py --img 640 --batch 16  --epochs 50 --data dataset.yaml --weights yolov5s.pt --device 0
'''

