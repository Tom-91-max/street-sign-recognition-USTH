from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8s.pt")   # hoặc yolov8n.pt
    model.train(
        data="configs/zalo.yaml",
        imgsz=960,
        epochs=100,
        batch=16,
        device=0,     # "cpu" nếu không có GPU
        project="runs",
        name="zalo_v8s_960",
        patience=20
    )
