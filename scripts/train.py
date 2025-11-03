# scripts/train.py

import os
import argparse
from datetime import datetime
import pandas as pd
from ultralytics import YOLO

# === ADDED BY BASELINE ENGINEER ==============================================
# Guard ép CPU khi --device cpu để tránh Ultralytics đọc settings cũ (device=0)
# và để phòng trường hợp bạn vô tình chạy bản file cũ.
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline YOLOv8n/s @img=960, epochs≈100 (Zalo traffic sign)"
    )
    parser.add_argument("--model", choices=["n", "s"], default="s")
    parser.add_argument("--img", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")          # 'cpu' hoặc 0,1,...
    parser.add_argument("--data", default="configs/zalo.yaml")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default=None)
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()

    # === ADDED BY BASELINE ENGINEER (FORCE CPU IF NEEDED) ====================
    # Nếu --device=cpu -> chặn CUDA hoàn toàn (kể cả settings cũ của Ultralytics)
    if str(args.device).lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""   # ép không dùng GPU
    # ========================================================================

    weights = "yolov8s.pt" if args.model == "s" else "yolov8n.pt"

    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.name = f"baseline_v8{args.model}_img{args.img}_e{args.epochs}_{timestamp}"

    model = YOLO(weights)  # GIỮ NGUYÊN tinh thần code gốc

    # === ADDED BY BASELINE ENGINEER (TRAIN) ==================================
    model.train(
        data=args.data,
        imgsz=args.img,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,         # <- nhận từ CLI
        project=args.project,
        name=args.name,
        patience=args.patience
    )
    # ========================================================================

    # === ADDED BY BASELINE ENGINEER (VALIDATE + PLOTS) ======================
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
    # ========================================================================

    # === ADDED BY BASELINE ENGINEER (SAVE SUMMARY CSV) ======================
    run_dir = os.path.join(args.project, args.name)
    os.makedirs(run_dir, exist_ok=True)

    summary = {
        "model": f"yolov8{args.model}.pt",
        "imgsz": args.img,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": str(args.device),
        "data": args.data,
        "project": args.project,
        "name": args.name,
        "run_dir": run_dir,
        "mAP50": mAP50,
        "mAP50_95": mAP5095,
        "precision": precision,
        "recall": recall,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(run_dir, "metrics_summary.csv"), index=False)

    print("\n=== BASELINE METRICS (val) ===")
    print(f"run_dir     : {run_dir}")
    print(f"mAP@50      : {mAP50:.4f}")
    print(f"mAP@50:95   : {mAP5095:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print("================================\n")
    # ========================================================================


if __name__ == "__main__":
    main()
