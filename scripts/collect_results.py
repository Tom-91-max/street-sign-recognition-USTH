# collect_results.py - Role E (MLOps) - auto short version
# Quét runs/*, đọc results.csv hoặc results.txt, gộp thành experiments.csv và vẽ mAP@50

import sys, re, argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# -- Hàm tiện: lấy cột nếu tồn tại, có fallback tên phổ biến
def get_col(row_or_df, names):
    for n in names:
        if isinstance(row_or_df, pd.DataFrame):
            if n in row_or_df.columns: return n
        else:
            if n in row_or_df.index: return n
    return None

# -- Đọc hàng cuối từ results.csv của YOLO
def read_results_csv(csv_path: Path):
    try:
        df = pd.read_csv(csv_path)
        if df.empty: return None
        row = df.iloc[-1]
        m50  = get_col(df, ["metrics/mAP50","mAP50","map50"])
        m95  = get_col(df, ["metrics/mAP50-95","mAP50:95","mAP50-95","map50-95"])
        prec = get_col(df, ["metrics/precision","precision"])
        rec  = get_col(df, ["metrics/recall","recall"])
        return {
            "mAP50": float(row[m50]) if m50 else None,
            "mAP5095": float(row[m95]) if m95 else None,
            "Precision": float(row[prec]) if prec else None,
            "Recall": float(row[rec]) if rec else None,
            "epochs": int(len(df)),
        }
    except Exception:
        return None

# -- Đọc số liệu cơ bản từ results.txt (nếu không có CSV)
def read_results_txt(txt_path: Path):
    try:
        t = txt_path.read_text(encoding="utf-8", errors="ignore")
        f = lambda pat: (re.search(pat, t, re.I) or [None])
        g = lambda m: float(m.group(1)) if m else None
        return {
            "mAP50":  g(re.search(r"mAP@?50\s*[:=]\s*([0-9.]+)", t, re.I)),
            "mAP5095":g(re.search(r"(?:mAP@?50[-: ]?95|mAP50[:\-]95)\s*[:=]\s*([0-9.]+)", t, re.I)),
            "Precision": g(re.search(r"precision\s*[:=]\s*([0-9.]+)", t, re.I)),
            "Recall":    g(re.search(r"recall\s*[:=]\s*([0-9.]+)", t, re.I)),
            "epochs": 0,
        }
    except Exception:
        return None

# -- Suy luận nhãn cơ bản từ tên thư mục run (model/imgsz/SAHI)
def infer_meta(run_dir: Path):
    name = run_dir.name.lower()
    model = "yolov8s"
    for m in ["yolov8n","yolov8s","yolov8m","yolov8l","yolov8x"]:
        if m in name: model = m; break
    imgsz = 1280 if "1280" in name else 960
    is_sahi = "sahi" in name
    return model, imgsz, ("Yes" if is_sahi else "No"), ("SAHI" if is_sahi else "Baseline")

def main(root="runs", out_dir="plots", out_csv="experiments.csv"):
    root = Path(root)
    if not root.exists():
        print(f"Error: root folder not found: {root}")
        sys.exit(1)

    rows = []
    # Ưu tiên results.csv
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
    # Fallback results.txt (nếu run nào không có CSV)
    for p in root.rglob("results.txt"):
        # bỏ qua nếu thư mục đã có csv
        if (p.parent / "results.csv").exists(): continue
        stats = read_results_txt(p)
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
        print("Warning: no results.csv or results.txt found. Nothing to aggregate.")
        sys.exit(0)

    df = pd.DataFrame(rows)
    # Lưu bảng đúng format đề bài
    cols = ["model","imgsz","epochs","augment","SAHI","mAP50","mAP5095","Precision","Recall","Note"]
    df[cols].to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved table to: {out_csv}")
    print("=== Experiment Summary ===")
    print(df[cols])

    # Vẽ biểu đồ mAP@50 theo Note (Baseline vs SAHI)
    Path(out_dir).mkdir(exist_ok=True)
    ax = df.plot(kind="bar", x="Note", y="mAP50", legend=False)
    ax.set_title("mAP@50 Comparison"); ax.set_xlabel("Configuration (Note)"); ax.set_ylabel("mAP@50")
    ax.get_figure().savefig(Path(out_dir)/"performance_comparison.png", bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {Path(out_dir)/'performance_comparison.png'}")

if __name__ == "__main__":
    # Tham số gọn: đủ dùng nhưng không rườm rà
    ap = argparse.ArgumentParser(description="Auto-collect YOLO results and plot (Role E)")
    ap.add_argument("--root", default="runs", help="Root to scan (e.g., runs/ or runs/detect/)")
    ap.add_argument("--out",  default="plots", help="Folder to save plots")
    ap.add_argument("--csv",  default="experiments.csv", help="Output CSV filename")
    a = ap.parse_args()
    main(a.root, a.out, a.csv)
