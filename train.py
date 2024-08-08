from ultralytics import YOLO

def main():    
    # Load a model
    model = YOLO("config/yolov8m-face.yaml").load("models/yolov8n.pt")  # build a new model from YAML

    # Train the model
    results = model.train(data="config/wider_face.yaml", epochs=120, imgsz=640, batch=16, device=0)

    print(results)

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()