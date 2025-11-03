# scripts/sahi_infer.py
# ======================================================
# Dự án: Street Sign Recognition (Zalo Dataset)
# ======================================================
from PIL import Image
import matplotlib.pyplot as plt
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os

def main():
    # === 1️⃣ Đường dẫn model YOLO tốt nhất (baseline) ===
    model_path = "runs/detect/baseline_v8s_img960_e100_20251103-131026/weights/best.pt"

    # === 2️⃣ Thư mục ảnh test hoặc val ===
    image_dir = "data/images/val"

    # === 3️⃣ Thư mục lưu kết quả inference của SAHI ===
    output_dir = "runs/sahi_vis"
    os.makedirs(output_dir, exist_ok=True)

    # === 4️⃣ Khởi tạo model với SAHI ===
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=0.3,
        device="cuda:0"  # nếu không có GPU thì đổi thành "cpu"
    )

    # === 5️⃣ Chạy inference SAHI trên từng ảnh ===
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

        # Lưu ảnh kết quả có bounding boxes
        result.export_visuals(export_dir=output_dir)

    print(f"\n✅ SAHI inference hoàn tất! Ảnh kết quả được lưu tại: {output_dir}")

if __name__ == "__main__":
    main()
    
# ===========================================
# 🔍 PHẦN 2: So sánh ảnh baseline vs SAHI
# ===========================================


def compare_baseline_sahi(baseline_dir="runs/pred_val", sahi_dir="runs/sahi_vis", output_path="runs/comparison"):
    os.makedirs(output_path, exist_ok=True)

    common_files = [f for f in os.listdir(sahi_dir) if f in os.listdir(baseline_dir)]
    if not common_files:
        print("⚠️ Không tìm thấy ảnh trùng tên giữa baseline và SAHI.")
        return

    for file in common_files:
        base_img = Image.open(os.path.join(baseline_dir, file))
        sahi_img = Image.open(os.path.join(sahi_dir, file))

        # Tạo hình hiển thị so sánh
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(base_img)
        axes[0].set_title("Baseline YOLOv8")
        axes[0].axis("off")

        axes[1].imshow(sahi_img)
        axes[1].set_title("SAHI Inference")
        axes[1].axis("off")

        plt.tight_layout()
        compare_path = os.path.join(output_path, f"compare_{file}")
        plt.savefig(compare_path)
        plt.close()

    print(f"✅ Đã tạo ảnh so sánh baseline vs SAHI tại: {output_path}")

# Gọi hàm sau khi chạy inference
if __name__ == "__main__":
    main()
    compare_baseline_sahi()
