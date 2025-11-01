# collect_results.py - Role E (MLOps)
# Đọc experiments.csv theo đúng format trong tài liệu và vẽ biểu đồ mAP@50

import pandas as pd
import matplotlib.pyplot as plt
import argparse, sys
from pathlib import Path

def main(csv_file="experiments.csv", out_dir="plots"):
    # Kiểm tra file CSV
    if not Path(csv_file).exists():
        print(f"Error: File not found {csv_file}")
        sys.exit(1)

    # Đọc dữ liệu (đúng cột theo tài liệu)
    df = pd.read_csv(csv_file)
    required = ["model","imgsz","epochs","augment","SAHI","mAP50","mAP5095","Precision","Recall","Note"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("Error: Missing columns:", ", ".join(missing))
        sys.exit(1)

    print("=== Experiment Summary ===")
    print(df[required])

    # Tạo thư mục output nếu chưa có
    Path(out_dir).mkdir(exist_ok=True)

    # Chọn trục X: ưu tiên Note (Baseline/SAHI). Nếu thiếu, fallback sang model.
    x_col = "Note" if "Note" in df.columns else "model"

    # Vẽ biểu đồ mAP@50
    ax = df.plot(kind="bar", x=x_col, y="mAP50", legend=False)
    ax.set_title("mAP@50 Comparison")
    ax.set_xlabel(x_col)
    ax.set_ylabel("mAP@50")

    # Lưu biểu đồ
    fig = ax.get_figure()
    out_path = Path(out_dir) / "performance_comparison.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved at: {out_path}")

if __name__ == "__main__":
    # Tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Compare model results (Role E)")
    parser.add_argument("--csv", default="experiments.csv", help="Path to CSV results file")
    parser.add_argument("--out", default="plots", help="Folder to save the plot")
    args = parser.parse_args()
    main(args.csv, args.out)
