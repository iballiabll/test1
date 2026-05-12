# -*- coding: utf-8 -*-
"""Verify runtime dependencies and project assets for the traffic demo."""

import argparse
import importlib
import json
import sys
from importlib import metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_PACKAGES = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "PIL": "pillow",
    "matplotlib": "matplotlib",
    "torch": "torch",
    "torchvision": "torchvision",
    "ultralytics": "ultralytics",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Verify project environment.")
    parser.add_argument("--load-model", action="store_true", help="Load YOLO weights for a stronger check.")
    return parser.parse_args()


def package_version(package_name):
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def check_imports():
    print("[1/5] 检查 Python 包")
    missing = []
    for module_name, package_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
            print(f"  OK  {package_name} {package_version(package_name)}")
        except Exception as exc:
            missing.append(package_name)
            print(f"  ERR {package_name}: {exc}")
    if missing:
        raise RuntimeError("缺少依赖: " + ", ".join(missing))


def check_config():
    print("[2/5] 检查配置文件")
    config_path = ROOT / "configs" / "traffic_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    print(f"  OK  {config_path.relative_to(ROOT)}")
    return config


def check_assets(config):
    print("[3/5] 检查模型与输出目录")
    model_path = ROOT / config.get("model_path", "yolov8n.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    print(f"  OK  模型文件: {model_path.relative_to(ROOT)}")

    output_dir = ROOT / config.get("output_dir", "检测结果")
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = output_dir / ".write_test"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink(missing_ok=True)
    print(f"  OK  输出目录可写: {output_dir.relative_to(ROOT)}")
    return model_path


def check_tkinter():
    print("[4/5] 检查 Tkinter GUI")
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    root.destroy()
    print("  OK  Tkinter 可创建窗口")


def check_model(model_path):
    print("[5/5] 检查 YOLO 权重加载")
    import torch
    from ultralytics import YOLO

    try:
        YOLO(str(model_path))
    except Exception as exc:
        if "weights_only" not in str(exc):
            raise
        original_torch_load = torch.load

        def torch_load_compat(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = torch_load_compat
        try:
            YOLO(str(model_path))
        finally:
            torch.load = original_torch_load
    print("  OK  YOLO 模型加载成功")


def main():
    args = parse_args()
    print("=============================================")
    print(" 路眼项目环境检查")
    print("=============================================")
    print(f"Python: {sys.version.split()[0]}")
    print(f"解释器: {sys.executable}")
    print(f"项目目录: {ROOT}")

    check_imports()
    config = check_config()
    model_path = check_assets(config)
    check_tkinter()
    if args.load_model:
        check_model(model_path)
    else:
        print("[5/5] 跳过模型加载检查，可追加 --load-model 进行完整验证")

    print("=============================================")
    print("环境检查通过")
    print("=============================================")


if __name__ == "__main__":
    main()
