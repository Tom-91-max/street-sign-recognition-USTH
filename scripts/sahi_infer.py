# ======================================================
# scripts/sahi_infer.py
# Dự án: Street Sign Recognition (Zalo Dataset)
# Nhiệm vụ: Cải thiện phát hiện vật thể nhỏ bằng SAHI
# ======================================================

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image
import matplotlib.pyplot as plt
import os, csv


# ------------------- PHẦN 1: CHẠY SAHI -------------------
def run_sahi_inference():
    # 🔹 Đường dẫn tới model YOLO tốt nhất mà thành viên C đã train xong
    model_path = "runs/detect/baseline_v8s_img960_e100_20251103-131026/weights/best.pt"

    # 🔹 Thư mục chứa ảnh test hoặc val (nếu chưa có test riêng thì dùng val)
    image_dir = "data/images/val"

    # 🔹 Thư mục để lưu kết quả ảnh sau khi chạy SAHI
    output_dir = "runs/sahi_vis"
    os.makedirs(output_dir, exist_ok=True)

    # 🔹 Khởi tạo model YOLO thông qua SAHI (tự động tải vào GPU nếu có)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",          # loại model dùng trong SAHI
        model_path=model_path,        # đường dẫn tới model YOLO đã huấn luyện
        confidence_threshold=0.3,     # ngưỡng tin cậy để lọc dự đoán
        device="cuda:0"               # dùng GPU 0, đổi thành "cpu" nếu không có GPU
    )

    # 🔹 Chạy dự đoán từng ảnh trong thư mục test/val
    for img_name in os.listdir(image_dir):
        # chỉ xử lý file ảnh (jpg, png,...)
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # đường dẫn đầy đủ tới từng ảnh
        img_path = os.path.join(image_dir, img_name)

        # 🧩 get_sliced_prediction: hàm chính của SAHI
        # chia ảnh lớn thành các lát nhỏ để nhận diện vật thể nhỏ tốt hơn
        result = get_sliced_prediction(
            img_path,
            detection_model,
            slice_height=512,             # chiều cao mỗi lát
            slice_width=512,              # chiều rộng mỗi lát
            overlap_height_ratio=0.2,     # phần chồng giữa các lát theo chiều cao
            overlap_width_ratio=0.2       # phần chồng giữa các lát theo chiều rộng
        )

        # Xuất ảnh kết quả (vẽ bounding box, label, confidence) ra thư mục runs/sahi_vis/
        result.export_visuals(export_dir=output_dir)

    print(f"\n✅ SAHI inference hoàn tất! Ảnh kết quả được lưu tại: {output_dir}")


# ---------------- PHẦN 2: SO SÁNH VỚI BASELINE ----------------
def compare_baseline_sahi(
    baseline_dir="runs/pred_val",   # thư mục ảnh baseline (YOLO thường)
    sahi_dir="runs/sahi_vis",       # thư mục ảnh SAHI (YOLO chia lát)
    output_path="runs/comparison"   # nơi lưu ảnh so sánh
):
    os.makedirs(output_path, exist_ok=True)

    # 🔹 Tìm các ảnh trùng tên giữa baseline và SAHI (để ghép cạnh nhau)
    common_files = [f for f in os.listdir(sahi_dir) if f in os.listdir(baseline_dir)]
    if not common_files:
        print("⚠️ Không tìm thấy ảnh trùng tên giữa baseline và SAHI.")
        return

    # 🔹 Với mỗi ảnh trùng tên, ghép 2 ảnh (baseline vs SAHI) thành 1 ảnh so sánh
    for file in common_files:
        base_img = Image.open(os.path.join(baseline_dir, file))
        sahi_img = Image.open(os.path.join(sahi_dir, file))

        # Dùng matplotlib để hiển thị song song 2 ảnh
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(base_img)
        axes[0].set_title("Baseline YOLOv8")  # ảnh gốc (YOLO thường)
        axes[0].axis("off")

        axes[1].imshow(sahi_img)
        axes[1].set_title("SAHI Inference")   # ảnh cải thiện (YOLO + SAHI)
        axes[1].axis("off")

        plt.tight_layout()
        compare_path = os.path.join(output_path, f"compare_{file}")
        plt.savefig(compare_path)
        plt.close()

    print(f"✅ Đã tạo ảnh so sánh baseline vs SAHI tại: {output_path}")


# ---------------- PHẦN 3: GHI KẾT QUẢ VÀO CSV ----------------
def log_experiment_to_csv(
    csv_path="experiments.csv",   # file lưu bảng kết quả thí nghiệm
    model="yolov8s",
    imgsz=960,
    epochs=100,
    augment="default",
    SAHI=True,
    mAP50=0.22635,                   #=== thay số thực tế khi có kết quả
    mAP5095=0.1254,
    Precision=0.49084,
    Recall=0.22743,
    note="SAHI"
):
):
    # Kiểm tra file CSV có tồn tại chưa — nếu chưa thì tạo header
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            # Header (tên cột)
            writer.writerow([
                "model", "imgsz", "epochs", "augment", "SAHI",
                "mAP50", "mAP5095", "Precision", "Recall", "Note"
            ])
        # Ghi thêm một dòng kết quả
        writer.writerow([
            model, imgsz, epochs, augment,
            "Yes" if SAHI else "No",
            mAP50, mAP5095, Precision, Recall, note
        ])
    print(f"✅ Đã ghi kết quả SAHI vào {csv_path}")


# ---------------- MAIN CHẠY TOÀN BỘ ----------------
if __name__ == "__main__":
    # 1️⃣ Chạy inference với SAHI
    run_sahi_inference()

    # 2️⃣ So sánh ảnh SAHI với baseline
    compare_baseline_sahi()

    # 3️⃣ Ghi kết quả mAP và các chỉ số vào bảng experiments.csv
    #    thay các giá trị mAP50, mAP5095, Precision, Recall theo kết quả thật
   log_experiment_to_csv(
        mAP50=0.22635,
        mAP5095=0.1254,
        Precision=0.49084,
        Recall=0.22743
    )
