from ultralytics import YOLO

# === ADDED BY BASELINE ENGINEER ==============================================
# - Thêm chế độ predict có CLI: --weights, --source, --img, --conf, --device,
#   --project, --name, --save_txt, --save_conf, --do_val, --data.
# - Ghi thêm file CSV: predictions.csv (image, class_id, class_name, conf, bbox).
# - Không sửa code cũ; nếu chạy với tham số CLI hoặc --be, chế độ mới sẽ chạy
#   và thoát trước khi vào khối main gốc.
# ============================================================================

# --- BE: imports phụ trợ (THÊM MỚI) ---
# (không ảnh hưởng code cũ)
import os
import sys
import argparse
import csv
from pathlib import Path


def _be_parse_args():
    p = argparse.ArgumentParser(
        description="[BE] YOLOv8 Predict with CSV export (non-invasive mode)"
    )
    p.add_argument("--weights", default="runs/detect/zalo_v8s_960/weights/best.pt",
                   help="Đường dẫn .pt để infer (mặc định: best.pt trong run cũ)")
    p.add_argument("--source", default="data/images/val",
                   help="Ảnh/thư mục/video để predict")
    p.add_argument("--img", type=int, default=960, help="imgsz (mặc định: 960)")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--device", default="cpu", help="'cpu' hoặc chỉ số GPU, vd: 0")
    p.add_argument("--project", default="runs/detect", help="thư mục project")
    p.add_argument("--name", default=None, help="tên run; nếu bỏ trống YOLO sẽ tự tạo")
    p.add_argument("--save_txt", action="store_true", help="lưu labels *.txt yolo format")
    p.add_argument("--save_conf", action="store_true", help="lưu confidence vào *.txt")
    p.add_argument("--do_val", action="store_true",
                   help="sau predict thì chạy val để có mAP/PR/CM")
    p.add_argument("--data", default="configs/zalo.yaml",
                   help="data.yaml dùng cho --do_val")
    p.add_argument("--be", action="store_true",
                   help="bật rõ chế độ Baseline Engineer (tùy chọn)")
    return p.parse_args()


def _be_should_run():
    # Nếu có bất kỳ tham số BE/CLI nào → chạy chế độ mới và thoát trước main cũ
    flags = {
        "--weights", "--source", "--img", "--conf", "--device",
        "--project", "--name", "--save_txt", "--save_conf",
        "--do_val", "--data", "--be"
    }
    return any(a in flags for a in sys.argv[1:])


def _be_write_predictions_csv(save_dir: Path, results_list):
    """
    Ghi predictions.csv gồm: image_path, pred_idx, class_id, class_name, conf, x1,y1,x2,y2
    """
    csv_path = save_dir / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "pred_idx", "class_id", "class_name",
                         "conf", "x1", "y1", "x2", "y2"])
        idx = 0
        for res in results_list:
            names = res.names if hasattr(res, "names") else {}
            img_path = str(getattr(res, "path", ""))
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
            cls = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else []
            conf = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []
            for i in range(len(cls)):
                c_id = int(cls[i])
                c_name = names.get(c_id, str(c_id))
                x1, y1, x2, y2 = xyxy[i].tolist()
                writer.writerow([img_path, idx, c_id, c_name,
                                 float(conf[i]), x1, y1, x2, y2])
                idx += 1
    return csv_path


def _be_main():
    args = _be_parse_args()

    # Tải model
    model = YOLO(args.weights)

    # Predict
    results = model.predict(
        source=args.source,
        imgsz=args.img,
        conf=args.conf,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=args.project,
        name=args.name
    )

    # Lấy thư mục lưu (YOLO 8 trả về list Results; dùng entry đầu tiên)
    try:
        save_dir = Path(results[0].save_dir)
    except Exception:
        # fallback: project/name
        save_dir = Path(args.project) / (args.name or "predict")

    # Ghi predictions.csv
    csv_path = _be_write_predictions_csv(save_dir, results)

    print(f"\n[BE] Saved annotated outputs to: {save_dir}")
    print(f"[BE] Predictions CSV: {csv_path}")

    # Tùy chọn: chạy val để có metrics/plots cho báo cáo
    if args.do_val:
        print("[BE] Running model.val(...) to produce metrics & plots ...")
        metrics = model.val(data=args.data, imgsz=args.img, device=args.device, plots=True)

        rd = getattr(metrics, "results_dict", {}) or {}
        def _get(rd, keys, default=0.0):
            for k in keys:
                if k in rd:
                    return float(rd[k])
            return default

        mAP50   = _get(rd, ["metrics/mAP50(B)", "metrics/mAP50"], 0.0)
        mAP5095 = _get(rd, ["metrics/mAP50-95(B)", "metrics/mAP50-95"], 0.0)
        precision = _get(rd, ["metrics/precision(B)", "metrics/precision"], 0.0)
        recall    = _get(rd, ["metrics/recall(B)", "metrics/recall"], 0.0)

        # Ghi tóm tắt ra CSV trong cùng thư mục predict
        import pandas as pd
        pd.DataFrame([{
            "weights": args.weights,
            "imgsz": args.img,
            "device": str(args.device),
            "mAP50": mAP50,
            "mAP50_95": mAP5095,
            "precision": precision,
            "recall": recall
        }]).to_csv(save_dir / "metrics_summary_predict.csv", index=False)

        print("\n[BE] === METRICS (val) ===")
        print(f"[BE] mAP@50   : {mAP50:.4f}")
        print(f"[BE] mAP@50:95: {mAP5095:.4f}")
        print(f"[BE] Precision: {precision:.4f}")
        print(f"[BE] Recall   : {recall:.4f}")
        print("[BE] ======================\n")


# Nếu chạy script với tham số CLI → kích hoạt chế độ BE và thoát sớm
if __name__ == "__main__" and _be_should_run():
    _be_main()
    sys.exit(0)
# ============================================================================


# ========================== CODE GỐC (KHÔNG SỬA) =============================
if __name__ == "__main__":
    model = YOLO("runs/detect/zalo_v8s_960/weights/best.pt")
    model.predict(
        source="data/images/val",
        conf=0.25,
        save=True,
        save_txt=True,    # lưu file kết quả .txt
        save_conf=True,   # lưu cả confidence
        project="runs",
        name="pred_val"
    )
    model.val(project="runs", name="val_v8s_960")
# ============================================================================

