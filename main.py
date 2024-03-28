from ultralytics import YOLO

if __name__ == '__main__':
# Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model

    model.train(data="config.yaml", epochs=400, imgsz=1480, batch=-1, device=[0])  # train the model



