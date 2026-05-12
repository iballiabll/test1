"""Evaluate a trained traffic-scene YOLO model and export metrics."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate traffic detection model.")
    parser.add_argument("--data", default="datasets/traffic/data.yaml", help="YOLO dataset yaml path.")
    parser.add_argument("--weights", default="runs/traffic/yolov8n_traffic/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="Examples: cpu, 0, 0,1")
    parser.add_argument("--output", default="检测结果/evaluation_metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.weights)
    data_path = Path(args.data)
    if not model_path.exists():
        raise FileNotFoundError(f"Weights not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    metrics = load_yolo(str(model_path)).val(
        data=str(data_path),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "weights": str(model_path),
        "data": str(data_path),
        "box_map50": float(metrics.box.map50),
        "box_map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr)
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def load_yolo(model_path):
    try:
        return YOLO(model_path)
    except Exception as exc:
        if "weights_only" not in str(exc) or not Path(model_path).exists():
            raise

        original_torch_load = torch.load

        def torch_load_compat(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = torch_load_compat
        try:
            return YOLO(model_path)
        finally:
            torch.load = original_torch_load


if __name__ == "__main__":
    main()
