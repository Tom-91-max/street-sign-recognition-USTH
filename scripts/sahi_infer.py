# scripts/sahi_infer.py
# ----------------------------------------------
# SAHI Inference Script for Small Object Detection Improvement


from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os

def main():
    # === 1️⃣ Đường dẫn model (best.pt) từ thành viên C ===
    model_path = "runs/detect/zalo_v8s_960/weights/best.pt"

    # === 2️⃣ Thư mục chứa ảnh test (bạn có thể đổi nếu cần) ===
    # Nếu team có thư mục test riêng:
    #   image_dir = "data/images/test"
    # Hoặc dùng ảnh dự đoán baseline của C để so sánh:
    image_dir = "runs/pred_val"

    # === 3️⃣ Thư mục lưu kết quả SAHI ===
    output_dir = "runs/sahi_vis"
    os.makedirs(output_dir, exist_ok=True)

    # === 4️⃣ Tạo model SAHI ===
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=0.3,
        device="cuda:0"  # đổi thành "cpu" nếu không có GPU
    )

    # === 5️⃣ Chạy SAHI trên từng ảnh test ===
    for img_name in os.listdir(image_dir):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue  # bỏ qua file không phải ảnh

        img_path = os.path.join(image_dir, img_name)

        result = get_sliced_prediction(
            img_path,
            detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )

        # Xuất ảnh kết quả
        result.export_visuals(export_dir=output_dir)

    print(f"\n✅ SAHI inference hoàn tất! Ảnh kết quả lưu tại: {output_dir}")

if __name__ == "__main__":
    main()
