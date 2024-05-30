# face-recognition

# detect

yolo detect train data=data-yolo.yaml model=yolov8n.yaml epochs=10 lr0=0.01

# classify

yolo classify train data=dataset-yolov8-cls model=yolov8n-cls.yaml epochs=10 lr0=0.01
