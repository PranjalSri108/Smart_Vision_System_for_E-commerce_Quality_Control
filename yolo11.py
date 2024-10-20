from ultralytics import YOLO

model = YOLO('yolo11n.pt')

model.train(
    data='/home/robosobo/GRID_6.0/Datasets/Bottle_Can_Quality.v1i.yolov11/data.yaml',  
    epochs=20,  
    batch=32,  
    imgsz=640, 
    name='freshness_yolo11'
)
