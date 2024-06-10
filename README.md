# face-recognition

# raw dataset: Data_Test, Data_Train, Data_Validation

# detection dataset: Data_Augmented => dataset-detection (Long's format), imgsz = 450

# classification dataset: Cropped_Face_Images => dataset-yolov8-cls (yolo format)

# detect yolov8

note: dataset-yolov8-detect.yaml must be placed at the root folder

`yolo detect train data=dataset-yolov8-detect.yaml model=yolov8n.yaml save_period=5 imgsz=640 epochs=10`

# train classify yolov8

`yolo task=classify mode=train data=dataset-yolov8-cls model=yolov8n-cls.yaml save_period=2 imgsz=120 epochs=10`

# detect Long's model

run in Face_Detection.ipynb with datasets/dataset-detection

# train classify Long's model

run in Face_Recognition.ipynb with datasets/dataset-yolov8-cls
