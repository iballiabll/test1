"""Train a traffic-scene YOLO model for the competition project."""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train traffic detection model.")
    parser.add_argument("--data", default="datasets/traffic/data.yaml", help="YOLO dataset yaml path.")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model or checkpoint.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/traffic")
    parser.add_argument("--name", default="yolov8n_traffic")
    parser.add_argument("--device", default=None, help="Examples: cpu, 0, 0,1")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    model = load_yolo(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device
    )


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
