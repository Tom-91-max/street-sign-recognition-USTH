from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/detect/zalo_v8s_960/weights/best.pt")
    model.predict(source="data/images/val", conf=0.25, save=True, project="runs", name="pred_val")
    model.val(project="runs", name="val_v8s_960")
