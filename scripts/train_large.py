# scripts/train_large.py
# =========================================
# Train YOLO với kích thước ảnh lớn hơn
# (Tùy chọn - để cải thiện phát hiện vật thể nhỏ)
# =========================================
from ultralytics import YOLO
import os

def main():
    model = YOLO("yolov8s.pt")

    model.train(
        data="configs/zalo.yaml",
        imgsz=1280,       # hoặc 1536 nếu máy mạnh
        epochs=100,
        batch=8,          # giảm batch size cho đỡ nặng GPU
        device=0,         # hoặc "cpu" nếu không có GPU
        project="runs/detect",
        name="large_img1280_e100",
        exist_ok=True
    )

    print("\n✅ Đã train xong mô hình YOLOv8 với ảnh lớn 1280px!")

if __name__ == "__main__":
    main()
