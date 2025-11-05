# scripts/collect_results.py - Role E (MLOps) - simplified version


import sys, argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# === Hàm tiện ích ===
def get_col(row_or_df, names):
    """Return the first existing column name found in the given list."""
    for n in names:
        if isinstance(row_or_df, pd.DataFrame):
            if n in row_or_df.columns:
                return n
        else:
            if n in row_or_df.index:
                return n
    return None


# === Đọc hàng cuối cùng từ results.csv ===
def read_results_csv(csv_path: Path):
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None

        row = df.iloc[-1]

        # ⚙️ Thêm hỗ trợ các tên cột có "(B)"
        m50 = get_col(df, ["metrics/mAP50", "metrics/mAP50(B)", "mAP50", "map50"])
        m95 = get_col(df, [
            "metrics/mAP50-95", "metrics/mAP50-95(B)",
            "mAP50:95", "mAP50-95", "map50-95"
        ])
        prec = get_col(df, ["metrics/precision", "metrics/precision(B)", "precision"])
        rec = get_col(df, ["metrics/recall", "metrics/recall(B)", "recall"])

        return {
            "mAP50": float(row[m50]) if m50 else None,
            "mAP5095": float(row[m95]) if m95 else None,
            "Precision": float(row[prec]) if prec else None,
            "Recall": float(row[rec]) if rec else None,
            "epochs": int(len(df)),
        }
    except Exception as e:
        print(f"⚠️ Error reading {csv_path}: {e}")
        return None


# === Suy luận thông tin từ tên thư mục run ===
def infer_meta(run_dir: Path):
    name = run_dir.name.lower()
    model = "yolov8s"
    for m in ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]:
        if m in name:
            model = m
            break
    imgsz = 1280 if "1280" in name else 960
    is_sahi = "sahi" in name
    return model, imgsz, ("Yes" if is_sahi else "No"), ("SAHI" if is_sahi else "Baseline")


# === Main ===
def main(root="runs", out_dir="plots", out_csv="experiments.csv"):
    root = Path(root)
    if not root.exists():
        print(f"❌ Folder not found: {root}")
        sys.exit(1)

    rows = []

    # Đọc mọi results.csv
    for p in root.rglob("results.csv"):
        stats = read_results_csv(p)
        if stats:
            model, imgsz, sahi, note = infer_meta(p.parent)
            rows.append({
                "model": model, "imgsz": imgsz, "epochs": stats["epochs"],
                "augment": "default", "SAHI": sahi,
                "mAP50": stats["mAP50"], "mAP5095": stats["mAP5095"],
                "Precision": stats["Precision"], "Recall": stats["Recall"],
                "Note": note
            })

    if not rows:
        print("⚠️ No results.csv files found.")
        sys.exit(0)

    df = pd.DataFrame(rows)

    # Lưu bảng tóm tắt
    cols = ["model", "imgsz", "epochs", "augment", "SAHI",
            "mAP50", "mAP5095", "Precision", "Recall", "Note"]
    df[cols].to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✅ Saved summary table to: {out_csv}")
    print(df[cols])

    # === Vẽ biểu đồ ===
    Path(out_dir).mkdir(exist_ok=True)
    try:
        ax = df.plot(kind="bar", x="Note", y="mAP50", legend=False)
        ax.set_title("mAP@50 Comparison")
        ax.set_xlabel("Configuration (Note)")
        ax.set_ylabel("mAP@50")
        fig_path = Path(out_dir) / "performance_comparison.png"
        ax.get_figure().savefig(fig_path, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved plot to: {fig_path}")
    except Exception as e:
        print(f"⚠️ Could not generate plot: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Collect YOLO results and generate summary + plots (CSV only)")
    ap.add_argument("--root", default="runs", help="Root folder to scan (e.g. runs/ or runs/detect/)")
    ap.add_argument("--out", default="plots", help="Folder to save plots")
    ap.add_argument("--csv", default="experiments.csv", help="Output CSV filename")
    args = ap.parse_args()
    main(args.root, args.out, args.csv)
