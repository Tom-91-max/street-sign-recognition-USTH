from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import os, glob, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="runs/detect/zalo_v8s_960/weights/best.pt")
parser.add_argument("--source", default="data/images/test")
parser.add_argument("--out", default="runs/sahi_vis")
parser.add_argument("--slice", type=int, default=512)
parser.add_argument("--overlap", type=float, default=0.2)
parser.add_argument("--device", default="cuda:0")  # "cpu" nếu không có GPU
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

dmodel = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=args.model,
    confidence_threshold=0.25,
    device=args.device
)

for img_path in glob.glob(f"{args.source}/*.*"):
    res = get_sliced_prediction(
        img_path, dmodel,
        slice_height=args.slice, slice_width=args.slice,
        overlap_height_ratio=args.overlap, overlap_width_ratio=args.overlap
    )
    res.export_visuals(export_dir=args.out)
