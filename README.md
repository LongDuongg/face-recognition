# face-recognition

# raw dataset: Data_Test, Data_Train, Data_Validation

# detection dataset: Data_Augmented => dataset-detection (Long's format)

# classification dataset: Cropped_Face_Images => dataset-yolov8-cls (yolo format)

# detect

yolo detect train data=data-yolo.yaml model=yolov8n.yaml epochs=10 lr0=0.01

# train classify

yolo task=classify mode=train data=dataset-yolov8-cls model=yolov8n-cls.yaml save_period=2 imgsz=120 epochs=10
