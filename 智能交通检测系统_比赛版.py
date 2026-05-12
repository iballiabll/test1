# -*- coding: utf-8 -*-
"""
智能交通检测系统 - 比赛专用版（增强版）
功能：
- 车辆检测（car, truck, bus, motorcycle, bicycle）
- 行人检测（person）
- 路障检测（stop sign, traffic light, cone等）
- 精确车速估算（基于距离和时间）
- 多目标跟踪
- 危险预警
- 结果保存
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import os
import json
import sys
from datetime import datetime
from collections import deque
import time
import math

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from ultralytics import YOLO
    import torch
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    print("✅ 所有依赖已就绪")
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("\n请运行:")
    print("pip install ultralytics opencv-python numpy pillow matplotlib torch")
    input("\n按回车键退出...")
    exit()


DEFAULT_CONFIG = {
    "model_path": "yolov8n.pt",
    "output_dir": "检测结果",
    "confidence": 0.3,
    "input": {
        "camera_index": 0,
        "camera_width": 1280,
        "camera_height": 720
    },
    "tracking": {
        "max_age": 30,
        "match_iou": 0.15,
        "max_center_distance": 120,
        "track_history": 20,
        "center_smoothing": 0.65,
        "min_confirmed_hits": 2
    },
    "detection": {
        "min_box_area_ratio": 0.00025,
        "class_thresholds": {
            "person": 0.35,
            "car": 0.35,
            "truck": 0.35,
            "bus": 0.35,
            "motorcycle": 0.35,
            "bicycle": 0.35,
            "traffic light": 0.55,
            "stop sign": 0.55,
            "fire hydrant": 0.55
        }
    },
    "roi": {
        "enabled": True,
        "show_overlay": True,
        "points": [[0.05, 0.55], [0.95, 0.55], [1.0, 1.0], [0.0, 1.0]]
    },
    "camera": {
        "focal_length": 800,
        "meters_per_pixel": None
    },
    "speed": {
        "speed_limit_kmh": 80,
        "max_valid_speed_kmh": 160,
        "max_speed_jump_kmh": 35,
        "smoothing_window": 8,
        "min_movement_px": 3.5,
        "stationary_speed_kmh": 3.0,
        "require_calibration": True,
        "stationary_frames": 5
    },
    "motion_ai": {
        "enabled": True,
        "stationary_flow_px": 0.45,
        "slow_flow_px": 1.4,
        "fast_flow_px": 4.0,
        "sample_step": 4,
        "ego_stationary_px": 0.8,
        "ego_driving_px": 2.0,
        "ego_turn_px": 0.35,
        "ego_shake_ratio": 0.65,
        "max_features": 400
    },
    "risk": {
        "vehicle_distance_m": 15,
        "pedestrian_distance_m": 20,
        "event_cooldown_sec": 3
    },
    "crossing_risk": {
        "enabled": True,
        "show_zone": True,
        "zone_points": [[0.18, 0.48], [0.82, 0.48], [0.95, 0.92], [0.05, 0.92]],
        "min_lateral_motion_px": 10,
        "near_vehicle_ratio": 0.28,
        "warn_levels": ["中", "高"]
    },
    "pedestrian_safety": {
        "enabled": True,
        "show_zone": True,
        "zone_points": [[0.0, 0.42], [1.0, 0.42], [1.0, 1.0], [0.0, 1.0]],
        "medium_area_ratio": 0.045,
        "high_area_ratio": 0.12,
        "approaching_size_change": 1.04,
        "fast_ai_index": 45,
        "front_band": [0.38, 0.62],
        "warn_levels": ["中", "高"]
    },
    "classes": {
        "vehicles": ["car", "truck", "bus", "motorcycle", "bicycle"],
        "barriers": ["stop sign", "traffic light", "fire hydrant"],
        "speed_objects": ["car", "truck", "bus", "motorcycle", "bicycle"]
    },
    "real_sizes": {
        "car": 4.5,
        "truck": 8.0,
        "bus": 12.0,
        "person": 1.7,
        "stop sign": 0.5,
        "traffic light": 0.6,
        "motorcycle": 2.0,
        "bicycle": 1.8
    },
    "class_names_cn": {
        "person": "行人",
        "bicycle": "自行车",
        "car": "小汽车",
        "motorcycle": "摩托车",
        "bus": "公交车",
        "truck": "卡车",
        "traffic light": "交通灯",
        "stop sign": "停车标志",
        "fire hydrant": "消防栓"
    }
}


def merge_config(base, override):
    """递归合并配置，保留默认字段。"""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path="configs/traffic_config.json"):
    if not os.path.exists(config_path):
        return DEFAULT_CONFIG
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        return merge_config(DEFAULT_CONFIG, user_config)
    except Exception as e:
        print(f"⚠️ 配置文件读取失败，使用默认配置: {e}")
        return DEFAULT_CONFIG


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0


class MotionAnalyzer:
    """基于光流和背景补偿的AI运动状态分析。"""
    def __init__(self, config):
        self.config = config
        self.prev_gray = None
        self.ego_state = {
            "status": "初始化",
            "motion_px": 0.0,
            "turn_score": 0.0,
            "confidence": 0.0,
            "features": 0
        }

    def reset(self):
        self.prev_gray = None
        self.ego_state = {
            "status": "初始化",
            "motion_px": 0.0,
            "turn_score": 0.0,
            "confidence": 0.0,
            "features": 0
        }

    def analyze(self, frame, tracked_objects):
        summary = {
            "moving_targets": 0,
            "stationary_targets": 0,
            "max_motion_index": 0,
            "ego_status": self.ego_state["status"],
            "ego_motion_px": self.ego_state["motion_px"],
            "ego_turn_score": self.ego_state["turn_score"],
            "ego_confidence": self.ego_state["confidence"]
        }
        if not self.config.get("enabled", True):
            return summary

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            for obj in tracked_objects.values():
                obj['motion_status'] = "AI初始化"
                obj['ai_speed_index'] = 0
            return summary

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            0.5,
            2,
            15,
            2,
            5,
            1.2,
            0
        )
        step = max(1, int(self.config.get("sample_step", 4)))
        background_mask = self.create_background_mask(gray.shape, tracked_objects)
        global_motion, ego_state = self.estimate_ego_motion(self.prev_gray, gray, flow, background_mask, step)
        self.ego_state = ego_state
        summary.update({
            "ego_status": ego_state["status"],
            "ego_motion_px": ego_state["motion_px"],
            "ego_turn_score": ego_state["turn_score"],
            "ego_confidence": ego_state["confidence"]
        })

        stationary_th = float(self.config.get("stationary_flow_px", 0.45))
        slow_th = float(self.config.get("slow_flow_px", 1.4))
        fast_th = float(self.config.get("fast_flow_px", 4.0))

        height, width = gray.shape
        for obj in tracked_objects.values():
            x1, y1, x2, y2 = map(int, obj['bbox'])
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height))
            if x2 <= x1 + 2 or y2 <= y1 + 2:
                obj['motion_status'] = "未知"
                obj['ai_speed_index'] = 0
                continue

            crop = flow[y1:y2:step, x1:x2:step]
            if crop.size == 0:
                obj['motion_status'] = "未知"
                obj['ai_speed_index'] = 0
                continue

            object_motion = np.median(crop.reshape(-1, 2), axis=0)
            relative_motion = object_motion - global_motion
            motion_mag = float(np.linalg.norm(relative_motion))
            index = int(max(0, min(100, round((motion_mag / max(fast_th, 0.01)) * 100))))

            if motion_mag < stationary_th:
                status = "静止"
                summary["stationary_targets"] += 1
                index = 0
            elif motion_mag < slow_th:
                status = "缓行"
                summary["moving_targets"] += 1
            elif motion_mag < fast_th:
                status = "行驶"
                summary["moving_targets"] += 1
            else:
                status = "快速"
                summary["moving_targets"] += 1

            obj['motion_status'] = status
            obj['motion_flow_px'] = motion_mag
            obj['ai_speed_index'] = index
            summary["max_motion_index"] = max(summary["max_motion_index"], index)

        self.prev_gray = gray
        return summary

    def create_background_mask(self, shape, tracked_objects):
        height, width = shape
        mask = np.full((height, width), 255, dtype=np.uint8)
        for obj in tracked_objects.values():
            x1, y1, x2, y2 = map(int, obj['bbox'])
            pad_x = max(8, int((x2 - x1) * 0.12))
            pad_y = max(8, int((y2 - y1) * 0.12))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(width, x2 + pad_x)
            y2 = min(height, y2 + pad_y)
            mask[y1:y2, x1:x2] = 0
        return mask

    def estimate_ego_motion(self, prev_gray, gray, flow, background_mask, step):
        sampled_flow = flow[::step, ::step]
        sampled_mask = background_mask[::step, ::step] > 0
        if np.any(sampled_mask):
            vectors = sampled_flow[sampled_mask].reshape(-1, 2)
        else:
            vectors = sampled_flow.reshape(-1, 2)

        global_motion = np.median(vectors, axis=0) if vectors.size else np.array([0.0, 0.0])
        motion_px = float(np.linalg.norm(global_motion))
        turn_score = 0.0
        confidence = 0.0

        points = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=int(self.config.get("max_features", 400)),
            qualityLevel=0.01,
            minDistance=8,
            mask=background_mask
        )
        if points is not None and len(points) >= 12:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None)
            if next_points is not None and status is not None:
                valid = status.reshape(-1) == 1
                prev_pts = points.reshape(-1, 2)[valid]
                next_pts = next_points.reshape(-1, 2)[valid]
                if len(prev_pts) >= 12:
                    transform, inliers = cv2.estimateAffinePartial2D(
                        prev_pts,
                        next_pts,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=3.0
                    )
                    if transform is not None:
                        dx = float(transform[0, 2])
                        dy = float(transform[1, 2])
                        motion_px = float(math.hypot(dx, dy))
                        angle_rad = math.atan2(transform[1, 0], transform[0, 0])
                        turn_score = abs(math.degrees(angle_rad))
                        confidence = float(np.mean(inliers)) if inliers is not None else 0.5

        status_text = self.classify_ego_state(motion_px, turn_score, confidence)
        return global_motion, {
            "status": status_text,
            "motion_px": motion_px,
            "turn_score": turn_score,
            "confidence": confidence,
            "features": int(0 if points is None else len(points))
        }

    def classify_ego_state(self, motion_px, turn_score, confidence):
        stationary_px = float(self.config.get("ego_stationary_px", 0.8))
        driving_px = float(self.config.get("ego_driving_px", 2.0))
        turn_px = float(self.config.get("ego_turn_px", 0.35))
        shake_ratio = float(self.config.get("ego_shake_ratio", 0.65))

        if motion_px < stationary_px:
            return "本车静止"
        if confidence < shake_ratio and motion_px >= driving_px:
            return "车身颠簸"
        if turn_score > turn_px:
            return "本车转向"
        if motion_px >= driving_px:
            return "本车行驶"
        return "本车缓动"


class VehicleTracker:
    """轻量多目标跟踪器：IOU优先，中心距离兜底。"""
    def __init__(self, max_age=30, match_iou=0.15, max_center_distance=120, track_history=20, speed_history=8, center_smoothing=0.65):
        self.vehicles = {}
        self.next_id = 0
        self.max_age = max_age
        self.match_iou = match_iou
        self.max_center_distance = max_center_distance
        self.track_history = track_history
        self.speed_history = speed_history
        self.center_smoothing = center_smoothing
        
    def update(self, detections):
        """更新跟踪"""
        new_vehicles = {}
        matched_ids = set()
        
        for det in detections:
            x1, y1, x2, y2, cls, conf = det
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            bbox = (x1, y1, x2, y2)
            
            best_id = None
            best_score = -float('inf')
            
            for vid, vdata in self.vehicles.items():
                if vid in matched_ids or vdata['age'] >= self.max_age:
                    continue

                # 不同类别之间不互相抢 ID，减少行人/车辆串号。
                if vdata['class'] != cls:
                    continue

                iou = bbox_iou(bbox, vdata['bbox'])
                dist = math.hypot(center[0] - vdata['center'][0], center[1] - vdata['center'][1])
                if iou < self.match_iou and dist > self.max_center_distance:
                    continue

                score = iou * 2 - dist / max(self.max_center_distance, 1)
                if score > best_score:
                    best_score = score
                    best_id = vid
            
            if best_id is not None:
                old_center = self.vehicles[best_id]['center']
                old_smooth_center = self.vehicles[best_id].get('smooth_center', old_center)
                old_bbox = self.vehicles[best_id]['bbox']
                
                alpha = self.center_smoothing
                smooth_center = (
                    old_smooth_center[0] * alpha + center[0] * (1 - alpha),
                    old_smooth_center[1] * alpha + center[1] * (1 - alpha)
                )
                pixel_movement = math.hypot(
                    smooth_center[0] - old_smooth_center[0],
                    smooth_center[1] - old_smooth_center[1]
                )
                
                old_area = (old_bbox[2] - old_bbox[0]) * (old_bbox[3] - old_bbox[1])
                new_area = (x2 - x1) * (y2 - y1)
                
                size_change = new_area / old_area if old_area > 0 else 1
                track = list(self.vehicles[best_id]['track']) + [smooth_center]
                
                new_vehicles[best_id] = {
                    'bbox': bbox,
                    'center': center,
                    'smooth_center': smooth_center,
                    'class': cls,
                    'conf': conf,
                    'age': 0,
                    'pixel_movement': pixel_movement,
                    'size_change': size_change,
                    'track': track[-self.track_history:],
                    'speed_history': self.vehicles[best_id]['speed_history'],
                    'speed': self.vehicles[best_id].get('speed', 0),
                    'distance': self.vehicles[best_id].get('distance', 0),
                    'hits': self.vehicles[best_id].get('hits', 1) + 1,
                    'stationary_count': self.vehicles[best_id].get('stationary_count', 0),
                    'is_stationary': self.vehicles[best_id].get('is_stationary', False)
                }
                matched_ids.add(best_id)
            else:
                new_vehicles[self.next_id] = {
                    'bbox': bbox,
                    'center': center,
                    'smooth_center': center,
                    'class': cls,
                    'conf': conf,
                    'age': 0,
                    'pixel_movement': 0,
                    'size_change': 1,
                    'track': [center],
                    'speed_history': deque(maxlen=self.speed_history),
                    'speed': 0,
                    'distance': 0,
                    'hits': 1,
                    'stationary_count': 0,
                    'is_stationary': False
                }
                self.next_id += 1
        
        for vid in self.vehicles:
            if vid not in new_vehicles:
                self.vehicles[vid]['age'] += 1
                if self.vehicles[vid]['age'] < self.max_age:
                    self.vehicles[vid]['pixel_movement'] = 0
                    self.vehicles[vid]['size_change'] = 1
                    new_vehicles[vid] = self.vehicles[vid]
        
        self.vehicles = new_vehicles
        return self.vehicles


class SmartTrafficSystem:
    """智能交通检测系统"""
    def __init__(self):
        self.window = tk.Tk()
        self.ui_thread_id = threading.get_ident()
        self.window.title("路眼")
        self.window.geometry("1600x950")
        self.window.configure(bg='#0f172a')
        self.window.state('zoomed')
        
        # 初始化
        self.config = load_config()
        self.label_font = self.load_font(20)
        self.panel_font = self.load_font(22)
        self.small_font = self.load_font(16)
        self.model = None
        self.cap = None
        self.running = False
        self.video_path = None
        self.source_type = "video"
        self.confidence = float(self.config.get("confidence", 0.3))
        self.model_path = self.config.get("model_path", "yolov8n.pt")
        self.tracker = self.create_tracker()
        
        # 真实物体尺寸（用于距离估算）
        self.real_sizes = self.config.get("real_sizes", DEFAULT_CONFIG["real_sizes"])
        
        # 相机参数（焦距，通过校准获得）
        self.camera_config = self.config.get("camera", {})
        self.input_config = self.config.get("input", {})
        self.detection_config = self.config.get("detection", {})
        self.roi_config = self.config.get("roi", {})
        self.motion_config = self.config.get("motion_ai", {})
        self.speed_config = self.config.get("speed", {})
        self.risk_config = self.config.get("risk", {})
        self.crossing_config = self.config.get("crossing_risk", {})
        self.pedestrian_safety_config = self.config.get("pedestrian_safety", {})
        self.class_config = self.config.get("classes", {})
        self.focal_length = float(self.camera_config.get("focal_length", 800))
        self.camera_index = int(self.input_config.get("camera_index", 0))
        self.camera_width = int(self.input_config.get("camera_width", 1280))
        self.camera_height = int(self.input_config.get("camera_height", 720))
        self.min_confirmed_hits = int(self.config.get("tracking", {}).get("min_confirmed_hits", 2))
        self.vehicle_classes = self.class_config.get("vehicles", DEFAULT_CONFIG["classes"]["vehicles"])
        self.barrier_classes = self.class_config.get("barriers", DEFAULT_CONFIG["classes"]["barriers"])
        self.speed_objects = self.class_config.get("speed_objects", DEFAULT_CONFIG["classes"]["speed_objects"])
        self.class_names_cn = self.config.get("class_names_cn", DEFAULT_CONFIG["class_names_cn"])
        self.event_last_seen = {}
        self.motion_analyzer = MotionAnalyzer(self.motion_config)
        self.demo_mode = "both"
        self.demo_mode_buttons = {}
        
        # 保存路径
        self.output_dir = self.config.get("output_dir", "检测结果")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 数据统计
        self.stats = {
            'total_vehicles': 0,
            'total_pedestrians': 0,
            'total_barriers': 0,
            'unique_vehicles': 0,
            'unique_pedestrians': 0,
            'unique_barriers': 0,
            'danger_count': 0,
            'max_speed': 0,
            'avg_speed': 0,
            'processing_fps': 0,
            'processed_frames': 0,
            'speed_calibrated': False,
            'max_motion_index': 0,
            'moving_targets': 0,
            'stationary_targets': 0,
            'ego_status': '初始化',
            'ego_motion_px': 0,
            'ego_confidence': 0,
            'crossing_risk_count': 0,
            'current_crossing_risks': 0,
            'high_crossing_risks': 0,
            'pedestrian_safety_risks': 0,
            'high_pedestrian_safety_risks': 0,
            'pedestrian_safety_alerts': 0,
            'invalid_speed_count': 0,
            'filtered_detections': 0,
            'stable_targets': 0,
            'trend_mode': 'motion_ai',
            'speed_records': deque(maxlen=100),
            'danger_events': [],
            'saved_video': None,
            'saved_screenshots': [],
            'report': None
        }
        self.seen_vehicle_ids = set()
        self.seen_pedestrian_ids = set()
        self.seen_barrier_ids = set()
        
        # 加载模型
        self.load_model()
        
        # 设置UI
        self.setup_ui()

    def load_font(self, size):
        candidates = [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\simsun.ttc",
            r"C:\Windows\Fonts\arial.ttf"
        ]
        for font_path in candidates:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, size)
                except Exception:
                    continue
        return ImageFont.load_default()

    def create_tracker(self):
        tracking = self.config.get("tracking", {})
        return VehicleTracker(
            max_age=int(tracking.get("max_age", 30)),
            match_iou=float(tracking.get("match_iou", 0.15)),
            max_center_distance=float(tracking.get("max_center_distance", 120)),
            track_history=int(tracking.get("track_history", 20)),
            speed_history=int(self.config.get("speed", {}).get("smoothing_window", 8)),
            center_smoothing=float(tracking.get("center_smoothing", 0.65))
        )

    def run_on_ui(self, callback, *args, **kwargs):
        if threading.get_ident() == self.ui_thread_id:
            callback(*args, **kwargs)
        else:
            self.window.after(0, lambda: callback(*args, **kwargs))
        
    def load_model(self):
        """加载模型"""
        device = '0' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {'GPU' if device == '0' else 'CPU'}")
        try:
            self.model = self.safe_load_yolo(self.model_path)
            print(f"✅ YOLOv8模型加载成功: {self.model_path}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            messagebox.showerror("错误", f"模型加载失败: {e}")

    def safe_load_yolo(self, model_path):
        """兼容 PyTorch 2.6+ 与旧版 ultralytics 的本地权重加载。"""
        try:
            return YOLO(model_path)
        except Exception as e:
            if "weights_only" not in str(e) or not os.path.exists(model_path):
                raise

            print("⚠️ 当前 PyTorch/ultralytics 组合触发 weights_only 兼容问题，按本地可信权重方式重试。")
            original_torch_load = torch.load

            def torch_load_compat(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return original_torch_load(*args, **kwargs)

            torch.load = torch_load_compat
            try:
                return YOLO(model_path)
            finally:
                torch.load = original_torch_load

    def create_metric_group(self, parent, title, accent_color, items):
        group = tk.LabelFrame(
            parent,
            text=f" {title} ",
            font=("Microsoft YaHei", 12, "bold"),
            bg='#172033',
            fg=accent_color,
            bd=0,
            labelanchor='nw'
        )
        group.pack(fill='x', pady=(0, 8))

        for idx, (label_text, key, color) in enumerate(items):
            row = idx // 2
            col = idx % 2
            cell = tk.Frame(group, bg='#0f172a', highlightthickness=1, highlightbackground='#263449')
            cell.grid(row=row, column=col, sticky='nsew', padx=4, pady=4)
            group.grid_columnconfigure(col, weight=1)

            tk.Label(
                cell,
                text=label_text,
                font=("Microsoft YaHei", 9),
                bg='#0f172a',
                fg='#94a3b8',
                anchor='w'
            ).pack(fill='x', padx=8, pady=(5, 0))

            value_label = tk.Label(
                cell,
                text="0",
                font=("Microsoft YaHei", 14, "bold"),
                bg='#0f172a',
                fg=color,
                anchor='w'
            )
            value_label.pack(fill='x', padx=8, pady=(0, 5))
            self.stat_labels[key] = value_label

    def create_right_panel(self, parent):
        shell = tk.Frame(parent, bg='#0f172a', width=480)
        self.right_panel_shell = shell
        shell.pack(side='right', fill='y', padx=(12, 0))
        shell.pack_propagate(False)

        canvas = tk.Canvas(shell, bg='#0f172a', highlightthickness=0)
        scrollbar = tk.Scrollbar(shell, orient='vertical', command=canvas.yview)
        content = tk.Frame(canvas, bg='#0f172a')
        content_id = canvas.create_window((0, 0), window=content, anchor='nw')

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        def on_content_configure(_event):
            canvas.configure(scrollregion=canvas.bbox('all'))

        def on_canvas_configure(event):
            canvas.itemconfigure(content_id, width=event.width)

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

        content.bind('<Configure>', on_content_configure)
        canvas.bind('<Configure>', on_canvas_configure)
        shell.bind('<Enter>', lambda _event: canvas.bind_all('<MouseWheel>', on_mousewheel))
        shell.bind('<Leave>', lambda _event: canvas.unbind_all('<MouseWheel>'))
        return content

    def create_demo_mode_selector(self, parent):
        selector = tk.Frame(parent, bg='#172033', highlightthickness=1, highlightbackground='#263449')
        selector.pack(fill='x', pady=(0, 10))

        header = tk.Frame(selector, bg='#172033')
        header.pack(fill='x', padx=10, pady=(9, 6))

        tk.Label(
            header,
            text="演示模式",
            font=("Microsoft YaHei", 12, "bold"),
            bg='#172033',
            fg='#e5e7eb',
            anchor='w'
        ).pack(side='left')

        self.demo_mode_label = tk.Label(
            header,
            text="综合展示",
            font=("Microsoft YaHei", 10, "bold"),
            bg='#0f172a',
            fg='#c4b5fd',
            padx=10,
            pady=3
        )
        self.demo_mode_label.pack(side='right')

        buttons = tk.Frame(selector, bg='#172033')
        buttons.pack(fill='x', padx=8, pady=(0, 9))

        mode_defs = [
            ("both", "综合展示", '#a78bfa'),
            ("vehicle", "车辆感知", '#38bdf8'),
            ("pedestrian", "行人避险", '#fb7185')
        ]
        for mode, text, color in mode_defs:
            btn = tk.Button(
                buttons,
                text=text,
                font=("Microsoft YaHei", 10, "bold"),
                bg='#0f172a',
                fg=color,
                activebackground='#1e293b',
                activeforeground='white',
                relief='flat',
                bd=0,
                padx=6,
                pady=7,
                command=lambda m=mode: self.set_demo_mode(m)
            )
            btn.pack(side='left', fill='x', expand=True, padx=3)
            self.demo_mode_buttons[mode] = btn

        self.refresh_demo_mode_ui()

    def refresh_demo_mode_ui(self):
        mode_meta = {
            "both": ("综合展示", '#a78bfa'),
            "vehicle": ("车辆感知", '#38bdf8'),
            "pedestrian": ("行人避险", '#fb7185')
        }
        active_text, active_color = mode_meta.get(self.demo_mode, mode_meta["both"])
        if hasattr(self, 'demo_mode_label'):
            self.demo_mode_label.config(text=active_text, fg=active_color)

        for mode, button in self.demo_mode_buttons.items():
            _, color = mode_meta.get(mode, mode_meta["both"])
            is_active = mode == self.demo_mode
            button.config(
                bg=color if is_active else '#0f172a',
                fg='#020617' if is_active else color
            )

    def set_demo_mode(self, mode):
        self.demo_mode = mode
        self.refresh_demo_mode_ui()
        mode_text = {
            "both": "综合展示模式：车辆感知与行人避险同时展示",
            "vehicle": "车辆感知模式：突出车辆、本车状态与AI速度指数",
            "pedestrian": "行人避险模式：突出行人视角危险来车与避让建议"
        }.get(mode, "综合展示模式")
        if hasattr(self, 'warning_text'):
            self.update_warning(f"已切换到{mode_text}")
    
    def setup_ui(self):
        """设置UI"""
        main_frame = tk.Frame(self.window, bg='#0f172a')
        main_frame.pack(fill='both', expand=True, padx=18, pady=14)
        
        # 标题
        title_frame = tk.Frame(main_frame, bg='#0f172a')
        title_frame.pack(fill='x', pady=(0, 14))
        
        title = tk.Label(
            title_frame,
            text="路眼：道路识别系统",
            font=("Microsoft YaHei", 28, "bold"),
            bg='#0f172a',
            fg='#f8fafc'
        )
        title.pack(side='left')
        
        subtitle = tk.Label(
            title_frame,
            text="实时感知  ·  多目标跟踪  ·  风险预警  ·  检测报告",
            font=("Microsoft YaHei", 12),
            bg='#0f172a',
            fg='#94a3b8'
        )
        subtitle.pack(side='left', padx=(18, 0), pady=(8, 0))
        
        # 中间区域
        center_frame = tk.Frame(main_frame, bg='#0f172a')
        center_frame.pack(fill='both', expand=True)

        # 右侧 - 数据面板先固定，避免视频图像请求尺寸挤压功能区
        right_frame = self.create_right_panel(center_frame)
        
        # 左侧 - 视频显示
        self.video_frame = tk.Frame(center_frame, bg='#111827', bd=0, highlightthickness=1, highlightbackground='#334155')
        self.video_frame.pack(side='left', fill='both', expand=True, padx=(0, 14))
        self.video_frame.pack_propagate(False)
        
        self.video_label = tk.Label(
            self.video_frame,
            text="📹 点击'上传视频'开始智能检测",
            font=("Microsoft YaHei", 16),
            bg='#111827',
            fg='#64748b'
        )
        self.video_label.pack(fill='both', expand=True)
        
        self.stat_labels = {}

        self.create_demo_mode_selector(right_frame)

        self.create_metric_group(
            right_frame,
            "车辆感知",
            '#38bdf8',
            [
                ("同帧车辆", 'total_vehicles', '#00ff88'),
                ("累计车辆", 'unique_vehicles', '#66ffaa'),
                ("本车状态", 'ego_status', '#34d399'),
                ("AI速度指数", 'max_motion_index', '#22d3ee'),
                ("静止车辆", 'stationary_targets', '#93c5fd'),
                ("稳定目标", 'stable_targets', '#a78bfa')
            ]
        )

        self.create_metric_group(
            right_frame,
            "行人避险",
            '#fb7185',
            [
                ("累计行人", 'unique_pedestrians', '#ffaa00'),
                ("行人避险", 'pedestrian_safety_risks', '#f43f5e'),
                ("横穿风险", 'current_crossing_risks', '#fb7185'),
                ("危险次数", 'danger_count', '#ff4444')
            ]
        )

        self.create_metric_group(
            right_frame,
            "系统状态",
            '#fbbf24',
            [
                ("累计路障", 'unique_barriers', '#ff66cc'),
                ("智能过滤", 'filtered_detections', '#fbbf24'),
                ("实时FPS", 'processing_fps', '#ffffff')
            ]
        )
        
        # 危险预警
        self.warning_frame = tk.LabelFrame(
            right_frame,
            text=" 风险事件 ",
            font=("Microsoft YaHei", 14, "bold"),
            bg='#172033',
            fg='#f97316',
            bd=0,
            labelanchor='nw'
        )
        self.warning_frame.pack(fill='x', pady=(0, 10))
        
        self.warning_text = tk.Text(
            self.warning_frame,
            height=5,
            font=("Microsoft YaHei", 10),
            bg='#0f172a',
            fg='#fed7aa',
            wrap='word'
        )
        self.warning_text.pack(fill='both', expand=True, padx=5, pady=5)
        self.warning_text.insert('end', "系统运行正常，暂无危险预警\n")
        self.warning_text.config(state='disabled')
        
        # 保存结果
        save_frame = tk.LabelFrame(
            right_frame,
            text=" 输出结果 ",
            font=("Microsoft YaHei", 14, "bold"),
            bg='#172033',
            fg='#22c55e',
            bd=0,
            labelanchor='nw'
        )
        save_frame.pack(fill='x', pady=(0, 10))
        
        self.save_path_label = tk.Label(
            save_frame,
            text="未保存",
            font=("Microsoft YaHei", 9),
            bg='#172033',
            fg='#94a3b8',
            wraplength=350,
            justify='left',
            anchor='w'
        )
        self.save_path_label.pack(fill='x', padx=10, pady=5)
        
        # 图表面板
        chart_frame = tk.LabelFrame(
            right_frame,
            text=" 运动趋势 ",
            font=("Microsoft YaHei", 14, "bold"),
            bg='#172033',
            fg='#38bdf8',
            bd=0,
            labelanchor='nw'
        )
        chart_frame.pack(fill='x', pady=(0, 10))
        
        self.fig = Figure(figsize=(4.2, 2.35), dpi=100, facecolor='#172033')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#0f172a')
        self.ax.tick_params(colors='#cbd5e1', labelsize=8)
        self.ax.set_xlabel('Frame', color='#cbd5e1', fontsize=8)
        self.ax.set_ylabel('AI index', color='#cbd5e1', fontsize=8)
        self.ax.set_xlim(0, 20)
        self.ax.set_ylim(0, 100)
        self.ax.grid(True, color='#334155', alpha=0.45, linewidth=0.8)
        for spine in self.ax.spines.values():
            spine.set_color('#475569')
        self.line, = self.ax.plot([], [], '#38bdf8', linewidth=2)
        self.chart_hint = self.ax.text(
            0.5,
            0.5,
            "Waiting for data",
            transform=self.ax.transAxes,
            ha='center',
            va='center',
            color='#94a3b8',
            fontsize=10
        )
        self.fig.tight_layout(pad=1.1)
        
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        chart_widget = self.canvas.get_tk_widget()
        chart_widget.configure(height=235)
        chart_widget.pack(fill='x', expand=False, padx=4, pady=4)
        self.canvas.draw()
        
        # 底部控制栏
        bottom_frame = tk.Frame(main_frame, bg='#0f172a')
        bottom_frame.pack(fill='x', pady=(14, 0))
        
        btn_frame = tk.Frame(bottom_frame, bg='#0f172a')
        btn_frame.pack(side='left')
        
        self.btn_upload = tk.Button(
            btn_frame,
            text="📤 上传视频",
            font=("Microsoft YaHei", 13, "bold"),
            width=12,
            bg='#0066cc',
            fg='white',
            activebackground='#0088ff',
            command=self.upload_video
        )
        self.btn_upload.pack(side='left', padx=5)

        self.btn_camera = tk.Button(
            btn_frame,
            text="🎥 实时摄像头",
            font=("Microsoft YaHei", 13, "bold"),
            width=12,
            bg='#008888',
            fg='white',
            activebackground='#00aaaa',
            command=self.start_camera_detection
        )
        self.btn_camera.pack(side='left', padx=5)
        
        self.btn_start = tk.Button(
            btn_frame,
            text="▶️ 开始检测",
            font=("Microsoft YaHei", 13, "bold"),
            width=12,
            bg='#00aa44',
            fg='white',
            activebackground='#00cc66',
            command=self.start_detection,
            state='disabled'
        )
        self.btn_start.pack(side='left', padx=5)
        
        self.btn_stop = tk.Button(
            btn_frame,
            text="⏹️ 停止",
            font=("Microsoft YaHei", 13, "bold"),
            width=12,
            bg='#cc0000',
            fg='white',
            activebackground='#ff0000',
            command=self.stop_detection,
            state='disabled'
        )
        self.btn_stop.pack(side='left', padx=5)
        
        self.btn_screenshot = tk.Button(
            btn_frame,
            text="📷 保存截图",
            font=("Microsoft YaHei", 13, "bold"),
            width=12,
            bg='#ff8800',
            fg='white',
            activebackground='#ffaa00',
            command=self.save_screenshot,
            state='disabled'
        )
        self.btn_screenshot.pack(side='left', padx=5)
        
        self.btn_report = tk.Button(
            btn_frame,
            text="📄 生成报告",
            font=("Microsoft YaHei", 13, "bold"),
            width=12,
            bg='#aa00ff',
            fg='white',
            activebackground='#cc44ff',
            command=self.generate_report,
            state='disabled'
        )
        self.btn_report.pack(side='left', padx=5)
        
        slider_frame = tk.Frame(bottom_frame, bg='#0f172a')
        slider_frame.pack(side='right', padx=20)
        
        tk.Label(
            slider_frame,
            text="灵敏度:",
            font=("Microsoft YaHei", 12),
            bg='#0f172a',
            fg='#cbd5e1'
        ).pack(side='left')
        
        self.conf_slider = tk.Scale(
            slider_frame,
            from_=10,
            to=90,
            orient='horizontal',
            length=180,
            command=self.on_slider_change,
            bg='#0f172a',
            fg='#38bdf8',
            troughcolor='#334155',
            highlightthickness=0
        )
        self.conf_slider.set(int(self.confidence * 100))
        self.conf_slider.pack(side='left', padx=10)
        
        self.conf_label = tk.Label(
            slider_frame,
            text=f"{int(self.confidence * 100)}%",
            font=("Microsoft YaHei", 12, "bold"),
            bg='#0f172a',
            fg='#38bdf8',
            width=5
        )
        self.conf_label.pack(side='left')
    
    def estimate_distance(self, bbox, class_name, frame_height):
        """基于检测框大小估算距离"""
        x1, y1, x2, y2 = bbox
        object_height_pixels = y2 - y1
        
        if class_name in self.real_sizes:
            real_height = self.real_sizes[class_name]
        else:
            real_height = 1.7
        
        if object_height_pixels > 0:
            distance = (real_height * self.focal_length) / object_height_pixels
            return min(distance, 100)
        return 50
    
    def calculate_speed(self, tracked_vehicle, fps, frame_height):
        """估算车速，并过滤明显不可信的跳变值。"""
        pixel_movement = tracked_vehicle['pixel_movement']
        bbox = tracked_vehicle['bbox']
        class_name = tracked_vehicle['class']

        if class_name not in self.speed_objects:
            return 0

        distance = self.estimate_distance(bbox, class_name, frame_height)
        tracked_vehicle['distance'] = distance

        history = tracked_vehicle['speed_history']
        previous_speed = float(np.mean(history)) if history else tracked_vehicle.get('speed', 0)
        min_movement = float(self.speed_config.get("min_movement_px", 3.5))
        stationary_speed = float(self.speed_config.get("stationary_speed_kmh", 3.0))
        stationary_frames = int(self.speed_config.get("stationary_frames", 5))
        calibrated_mpp = self.camera_config.get("meters_per_pixel")
        require_calibration = bool(self.speed_config.get("require_calibration", True))

        if require_calibration and not calibrated_mpp:
            history.clear()
            tracked_vehicle['speed'] = 0
            motion_status = tracked_vehicle.get('motion_status', "AI分析中")
            tracked_vehicle['is_stationary'] = motion_status == "静止"
            tracked_vehicle['speed_status'] = motion_status
            return 0

        if pixel_movement < min_movement or fps <= 0:
            history.append(0)
            tracked_vehicle['stationary_count'] = tracked_vehicle.get('stationary_count', 0) + 1
            if tracked_vehicle['stationary_count'] >= stationary_frames:
                history.clear()
                tracked_vehicle['is_stationary'] = True
                tracked_vehicle['speed'] = 0
                tracked_vehicle['speed_status'] = "静止"
                return 0
            tracked_vehicle['speed'] = 0 if previous_speed < stationary_speed * 2 else previous_speed * 0.5
            tracked_vehicle['speed_status'] = "低速"
            return tracked_vehicle['speed']
        else:
            tracked_vehicle['stationary_count'] = 0
            tracked_vehicle['is_stationary'] = False

        if calibrated_mpp:
            meters_per_pixel = float(calibrated_mpp)
        else:
            meters_per_pixel = distance / max(self.focal_length, 1)

        raw_speed = pixel_movement * meters_per_pixel * fps * 3.6
        max_valid_speed = float(self.speed_config.get("max_valid_speed_kmh", 160))
        max_jump = float(self.speed_config.get("max_speed_jump_kmh", 35))

        if raw_speed > max_valid_speed:
            self.stats['invalid_speed_count'] += 1
            return previous_speed

        if previous_speed > 0 and abs(raw_speed - previous_speed) > max_jump:
            raw_speed = previous_speed + math.copysign(max_jump, raw_speed - previous_speed)

        if raw_speed < stationary_speed:
            raw_speed = 0

        history.append(raw_speed)
        speed = float(np.mean(history))
        if speed < stationary_speed:
            speed = 0
        tracked_vehicle['speed_status'] = "运动" if speed > 0 else "静止"
        tracked_vehicle['speed'] = speed
        return speed

    def update_vehicle_metrics(self, tracked_vehicles, fps, frame_height):
        speeds = []
        for vehicle in tracked_vehicles.values():
            vehicle['distance'] = self.estimate_distance(vehicle['bbox'], vehicle['class'], frame_height)
            speed = self.calculate_speed(vehicle, fps, frame_height)
            vehicle['speed'] = speed
            if speed > 0:
                speeds.append(speed)
        return speeds

    def update_unique_counts(self, tracked_objects):
        stable_targets = 0
        for vid, obj in tracked_objects.items():
            if obj.get('age', 0) > 0 or not self.is_confirmed_track(obj):
                continue

            stable_targets += 1
            class_name = obj['class']
            if class_name in self.vehicle_classes:
                self.seen_vehicle_ids.add(vid)
            elif class_name == 'person':
                self.seen_pedestrian_ids.add(vid)
            elif class_name in self.barrier_classes:
                self.seen_barrier_ids.add(vid)

        self.stats['unique_vehicles'] = len(self.seen_vehicle_ids)
        self.stats['unique_pedestrians'] = len(self.seen_pedestrian_ids)
        self.stats['unique_barriers'] = len(self.seen_barrier_ids)
        self.stats['stable_targets'] = stable_targets

    def get_class_display_name(self, class_name):
        return self.class_names_cn.get(class_name, class_name)

    def get_risk_level(self, class_name, distance, speed):
        speed_limit = float(self.speed_config.get("speed_limit_kmh", 80))
        vehicle_distance = float(self.risk_config.get("vehicle_distance_m", 15))
        pedestrian_distance = float(self.risk_config.get("pedestrian_distance_m", 20))

        if class_name in ['car', 'truck', 'bus']:
            if speed >= speed_limit * 1.25 or distance <= vehicle_distance * 0.6:
                return "高"
            if speed > speed_limit or distance < vehicle_distance:
                return "中"
        if class_name == 'person' and distance < pedestrian_distance:
            return "高" if distance <= pedestrian_distance * 0.5 else "中"
        return "低"

    def is_confirmed_track(self, tracked_object):
        return tracked_object.get('hits', 0) >= self.min_confirmed_hits

    def get_roi_points(self, width, height):
        points = self.roi_config.get("points", DEFAULT_CONFIG["roi"]["points"])
        return np.array(
            [(int(x * width), int(y * height)) for x, y in points],
            dtype=np.int32
        )

    def is_in_roi(self, bbox, width, height):
        if not self.roi_config.get("enabled", True):
            return True
        x1, y1, x2, y2 = bbox
        foot_point = ((x1 + x2) / 2, y2)
        roi_points = self.get_roi_points(width, height)
        return cv2.pointPolygonTest(roi_points, foot_point, False) >= 0

    def get_crossing_zone_points(self, width, height):
        points = self.crossing_config.get("zone_points", DEFAULT_CONFIG["crossing_risk"]["zone_points"])
        return np.array(
            [(int(x * width), int(y * height)) for x, y in points],
            dtype=np.int32
        )

    def get_pedestrian_safety_zone_points(self, width, height):
        points = self.pedestrian_safety_config.get("zone_points", DEFAULT_CONFIG["pedestrian_safety"]["zone_points"])
        return np.array(
            [(int(x * width), int(y * height)) for x, y in points],
            dtype=np.int32
        )

    def is_in_pedestrian_safety_zone(self, bbox, width, height):
        if not self.pedestrian_safety_config.get("enabled", True):
            return False
        x1, y1, x2, y2 = bbox
        foot_point = ((x1 + x2) / 2, y2)
        zone_points = self.get_pedestrian_safety_zone_points(width, height)
        return cv2.pointPolygonTest(zone_points, foot_point, False) >= 0

    def get_vehicle_approach_direction(self, bbox, frame_width):
        x1, _, x2, _ = bbox
        center_ratio = ((x1 + x2) / 2) / max(frame_width, 1)
        front_band = self.pedestrian_safety_config.get("front_band", [0.38, 0.62])
        if center_ratio < float(front_band[0]):
            return "左侧来车", "向右避让"
        if center_ratio > float(front_band[1]):
            return "右侧来车", "向左避让"
        return "正前方来车", "暂停观察/后退"

    def is_in_crossing_zone(self, bbox, width, height):
        if not self.crossing_config.get("enabled", True):
            return False
        x1, y1, x2, y2 = bbox
        foot_point = ((x1 + x2) / 2, y2)
        zone_points = self.get_crossing_zone_points(width, height)
        return cv2.pointPolygonTest(zone_points, foot_point, False) >= 0

    def compute_pedestrian_crossing_risks(self, tracked_objects, frame_width, frame_height):
        if not self.crossing_config.get("enabled", True):
            self.stats['current_crossing_risks'] = 0
            self.stats['high_crossing_risks'] = 0
            return

        vehicles = [
            (vid, obj) for vid, obj in tracked_objects.items()
            if obj.get('class') in self.vehicle_classes and self.is_confirmed_track(obj)
        ]
        current_risks = 0
        high_risks = 0
        min_lateral_motion = float(self.crossing_config.get("min_lateral_motion_px", 10))
        near_vehicle_ratio = float(self.crossing_config.get("near_vehicle_ratio", 0.28))
        frame_diag = max(math.hypot(frame_width, frame_height), 1)

        for pid, person in tracked_objects.items():
            if person.get('class') != 'person' or not self.is_confirmed_track(person):
                continue

            in_zone = self.is_in_crossing_zone(person['bbox'], frame_width, frame_height)
            track = person.get('track', [])
            lateral_motion = abs(track[-1][0] - track[0][0]) if len(track) >= 2 else 0
            is_crossing = in_zone and lateral_motion >= min_lateral_motion

            px1, py1, px2, py2 = person['bbox']
            person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
            nearest_vehicle_id = None
            nearest_ratio = 1.0
            nearest_vehicle_motion = "未知"

            for vid, vehicle in vehicles:
                vx1, vy1, vx2, vy2 = vehicle['bbox']
                vehicle_center = ((vx1 + vx2) / 2, (vy1 + vy2) / 2)
                distance_ratio = math.hypot(
                    person_center[0] - vehicle_center[0],
                    person_center[1] - vehicle_center[1]
                ) / frame_diag
                if distance_ratio < nearest_ratio:
                    nearest_ratio = distance_ratio
                    nearest_vehicle_id = vid
                    nearest_vehicle_motion = vehicle.get('motion_status', '未知')

            vehicle_near = nearest_vehicle_id is not None and nearest_ratio <= near_vehicle_ratio
            ego_moving = self.stats.get('ego_status') not in ['初始化', '本车静止']
            vehicle_moving = nearest_vehicle_motion in ['缓行', '行驶', '快速'] or ego_moving

            score = 0
            reasons = []
            if in_zone:
                score += 1
                reasons.append("进入横穿区域")
            if is_crossing:
                score += 1
                reasons.append("存在横向运动")
            if vehicle_near:
                score += 1
                reasons.append("附近有车辆")
            if vehicle_near and vehicle_moving:
                score += 1
                reasons.append("车辆/本车处于运动状态")

            if score >= 4:
                level = "高"
            elif score >= 2:
                level = "中"
            elif score >= 1:
                level = "低"
            else:
                level = "无"

            person['crossing_risk_level'] = level
            person['crossing_reasons'] = reasons
            person['crossing_lateral_px'] = lateral_motion
            person['crossing_nearest_vehicle'] = nearest_vehicle_id
            person['crossing_vehicle_ratio'] = nearest_ratio

            if level in ["中", "高"]:
                current_risks += 1
            if level == "高":
                high_risks += 1

        self.stats['current_crossing_risks'] = current_risks
        self.stats['high_crossing_risks'] = high_risks

    def compute_pedestrian_safety_risks(self, tracked_objects, frame_width, frame_height):
        if not self.pedestrian_safety_config.get("enabled", True):
            self.stats['pedestrian_safety_risks'] = 0
            self.stats['high_pedestrian_safety_risks'] = 0
            return

        current_risks = 0
        high_risks = 0
        frame_area = max(frame_width * frame_height, 1)
        medium_area_ratio = float(self.pedestrian_safety_config.get("medium_area_ratio", 0.045))
        high_area_ratio = float(self.pedestrian_safety_config.get("high_area_ratio", 0.12))
        approaching_size_change = float(self.pedestrian_safety_config.get("approaching_size_change", 1.04))
        fast_ai_index = int(self.pedestrian_safety_config.get("fast_ai_index", 45))

        for vid, obj in tracked_objects.items():
            if obj.get('class') not in self.vehicle_classes or not self.is_confirmed_track(obj):
                continue

            bbox = obj['bbox']
            x1, y1, x2, y2 = bbox
            area_ratio = max((x2 - x1) * (y2 - y1), 0) / frame_area
            in_zone = self.is_in_pedestrian_safety_zone(bbox, frame_width, frame_height)
            direction, advice = self.get_vehicle_approach_direction(bbox, frame_width)
            motion_status = obj.get('motion_status', '未知')
            ai_index = int(obj.get('ai_speed_index', 0))
            size_change = float(obj.get('size_change', 1))
            approaching = size_change >= approaching_size_change or ai_index >= fast_ai_index or motion_status in ['行驶', '快速']

            score = 0
            reasons = []
            if in_zone:
                score += 1
                reasons.append("进入行人前方危险区")
            if area_ratio >= medium_area_ratio:
                score += 1
                reasons.append("车辆距离较近")
            if area_ratio >= high_area_ratio:
                score += 1
                reasons.append("车辆占比过大")
            if approaching:
                score += 1
                reasons.append("存在接近趋势")
            if direction == "正前方来车":
                score += 1
                reasons.append("位于正前方")

            if score >= 4:
                level = "高"
            elif score >= 2:
                level = "中"
            elif score >= 1:
                level = "低"
            else:
                level = "无"

            obj['pedestrian_safety_level'] = level
            obj['pedestrian_safety_direction'] = direction
            obj['pedestrian_safety_advice'] = advice
            obj['pedestrian_safety_reasons'] = reasons
            obj['pedestrian_safety_area_ratio'] = area_ratio
            obj['pedestrian_safety_approaching'] = approaching

            if level in ["中", "高"]:
                current_risks += 1
            if level == "高":
                high_risks += 1

        self.stats['pedestrian_safety_risks'] = current_risks
        self.stats['high_pedestrian_safety_risks'] = high_risks

    def passes_detection_filters(self, class_name, conf, bbox, frame_width, frame_height):
        class_thresholds = self.detection_config.get("class_thresholds", {})
        min_conf = float(class_thresholds.get(class_name, self.confidence))
        if conf < min_conf:
            return False

        x1, y1, x2, y2 = bbox
        frame_area = max(frame_width * frame_height, 1)
        box_area_ratio = max((x2 - x1) * (y2 - y1), 0) / frame_area
        min_box_area_ratio = float(self.detection_config.get("min_box_area_ratio", 0.00025))
        if box_area_ratio < min_box_area_ratio:
            return False

        return self.is_in_roi(bbox, frame_width, frame_height)
    
    def upload_video(self):
        """上传视频"""
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            self.video_path = file_path
            self.source_type = "video"
            self.btn_start.config(state='normal')
            filename = os.path.basename(file_path)
            self.update_warning(f"✅ 已加载: {filename}")
            self.save_path_label.config(text=f"📂 输出目录: {self.output_dir}")

    def start_camera_detection(self):
        """启动摄像头实时检测。"""
        if self.running:
            return

        self.source_type = "camera"
        self.video_path = None
        self.update_warning(f"🎥 准备打开摄像头 {self.camera_index} 进行实时检测")
        self.start_detection()

    def get_source_label(self):
        if self.source_type == "camera":
            return f"摄像头 {self.camera_index}"
        return self.video_path or "未选择"
    
    def on_slider_change(self, value):
        """滑块改变"""
        self.confidence = int(value) / 100
        self.conf_label.config(text=f"{value}%")
    
    def update_warning(self, message, is_danger=False, event_key=None):
        """更新警告信息"""
        if threading.get_ident() != self.ui_thread_id:
            self.run_on_ui(self.update_warning, message, is_danger, event_key)
            return False

        if is_danger and event_key:
            now = time.monotonic()
            cooldown = float(self.risk_config.get("event_cooldown_sec", 3))
            last_seen = self.event_last_seen.get(event_key, 0)
            if now - last_seen < cooldown:
                return False
            self.event_last_seen[event_key] = now

        self.warning_text.config(state='normal')
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = "🚨" if is_danger else "ℹ️"
        self.warning_text.insert('end', f"[{timestamp}] {prefix} {message}\n")
        self.warning_text.see('end')
        self.warning_text.config(state='disabled')
        
        if is_danger:
            self.stats['danger_count'] += 1
            self.stat_labels['danger_count'].config(text=str(self.stats['danger_count']))
            self.stats['danger_events'].append({
                'time': timestamp,
                'event_key': event_key,
                'message': message
            })
        return True
    
    def start_detection(self):
        """开始检测"""
        if self.source_type != "camera" and not self.video_path:
            messagebox.showwarning("提示", "请先上传视频")
            return
        if self.model is None:
            messagebox.showerror("错误", "模型尚未加载成功，请先检查 yolov8n.pt 和依赖环境。")
            self.update_warning("❌ 模型尚未加载成功，无法开始检测")
            return
        
        self.running = True
        self.btn_upload.config(state='disabled')
        self.btn_camera.config(state='disabled')
        self.btn_start.config(state='disabled')
        self.btn_stop.config(state='normal')
        self.btn_screenshot.config(state='normal')
        self.btn_report.config(state='disabled')
        
        # 重置统计
        self.stats['total_vehicles'] = 0
        self.stats['total_pedestrians'] = 0
        self.stats['total_barriers'] = 0
        self.stats['unique_vehicles'] = 0
        self.stats['unique_pedestrians'] = 0
        self.stats['unique_barriers'] = 0
        self.stats['danger_count'] = 0
        self.stats['max_speed'] = 0
        self.stats['avg_speed'] = 0
        self.stats['processing_fps'] = 0
        self.stats['processed_frames'] = 0
        self.stats['speed_calibrated'] = bool(self.camera_config.get("meters_per_pixel"))
        self.stats['max_motion_index'] = 0
        self.stats['moving_targets'] = 0
        self.stats['stationary_targets'] = 0
        self.stats['ego_status'] = '初始化'
        self.stats['ego_motion_px'] = 0
        self.stats['ego_confidence'] = 0
        self.stats['crossing_risk_count'] = 0
        self.stats['current_crossing_risks'] = 0
        self.stats['high_crossing_risks'] = 0
        self.stats['pedestrian_safety_risks'] = 0
        self.stats['high_pedestrian_safety_risks'] = 0
        self.stats['pedestrian_safety_alerts'] = 0
        self.stats['invalid_speed_count'] = 0
        self.stats['filtered_detections'] = 0
        self.stats['stable_targets'] = 0
        self.stats['trend_mode'] = 'speed' if self.stats['speed_calibrated'] else 'motion_ai'
        self.stats['speed_records'].clear()
        self.stats['danger_events'].clear()
        self.stats['saved_video'] = None
        self.stats['saved_screenshots'].clear()
        self.stats['report'] = None
        self.seen_vehicle_ids.clear()
        self.seen_pedestrian_ids.clear()
        self.seen_barrier_ids.clear()
        self.event_last_seen.clear()
        self.tracker = self.create_tracker()
        self.motion_analyzer.reset()
        
        self.update_warning(f"🚀 开始智能检测，输入源：{self.get_source_label()}")
        
        # 启动检测线程
        self.detect_thread = threading.Thread(target=self.detect_video)
        self.detect_thread.daemon = True
        self.detect_thread.start()
    
    def detect_video(self):
        """检测视频"""
        out = None
        video_path = None
        frame_count = 0
        failed = False

        try:
            if self.source_type == "camera":
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            else:
                self.cap = cv2.VideoCapture(self.video_path)

            if not self.cap.isOpened():
                failed = True
                self.update_warning(f"❌ 无法打开输入源：{self.get_source_label()}")
                return

            fps = self.cap.get(cv2.CAP_PROP_FPS)
            fps = fps if fps and fps > 0 else 30
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                width, height = self.camera_width, self.camera_height

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_prefix = "摄像头检测视频" if self.source_type == "camera" else "检测视频"
            video_filename = f"{video_prefix}_{timestamp}.mp4"
            video_path = os.path.join(self.output_dir, video_filename)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            if not out.isOpened():
                self.update_warning("⚠️ 检测视频写入器打开失败，本次仅显示画面和保存截图")
                out = None
                video_path = None

            self.stats['saved_video'] = video_path

            screenshot_interval = 60

            while self.running and self.cap and self.cap.isOpened():
                frame_start = time.perf_counter()
                ret, frame = self.cap.read()
                if not ret:
                    break

                results = self.model(frame, conf=self.confidence, iou=0.45)

                detections = []
                vehicle_count = 0
                pedestrian_count = 0
                barrier_count = 0

                for box in results[0].boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    class_name = self.model.names[cls]

                    if not self.passes_detection_filters(class_name, conf, (x1, y1, x2, y2), width, height):
                        self.stats['filtered_detections'] += 1
                        continue

                    detections.append([x1, y1, x2, y2, class_name, conf])

                    if class_name in self.vehicle_classes:
                        vehicle_count += 1
                    elif class_name == 'person':
                        pedestrian_count += 1
                    elif class_name in self.barrier_classes:
                        barrier_count += 1

                tracked_vehicles = self.tracker.update(detections)
                motion_summary = self.motion_analyzer.analyze(frame, tracked_vehicles)
                self.stats['moving_targets'] = motion_summary['moving_targets']
                self.stats['stationary_targets'] = motion_summary['stationary_targets']
                self.stats['max_motion_index'] = max(self.stats['max_motion_index'], motion_summary['max_motion_index'])
                self.stats['ego_status'] = motion_summary['ego_status']
                self.stats['ego_motion_px'] = motion_summary['ego_motion_px']
                self.stats['ego_confidence'] = motion_summary['ego_confidence']
                speeds = self.update_vehicle_metrics(tracked_vehicles, fps, height)
                self.update_unique_counts(tracked_vehicles)
                self.compute_pedestrian_crossing_risks(tracked_vehicles, width, height)
                self.compute_pedestrian_safety_risks(tracked_vehicles, width, height)
                annotated_frame = self.draw_results(frame, tracked_vehicles, height)

                if out is not None:
                    out.write(annotated_frame)

                frame_count += 1
                if frame_count % screenshot_interval == 0:
                    screenshot_path = os.path.join(
                        self.output_dir,
                        f"截图_{timestamp}_{frame_count}.jpg"
                    )
                    cv2.imwrite(screenshot_path, annotated_frame)
                    self.stats['saved_screenshots'].append(screenshot_path)

                self.stats['total_vehicles'] = max(self.stats['total_vehicles'], vehicle_count)
                self.stats['total_pedestrians'] = max(self.stats['total_pedestrians'], pedestrian_count)
                self.stats['total_barriers'] = max(self.stats['total_barriers'], barrier_count)

                if speeds:
                    avg_speed = np.mean(speeds)
                    max_speed = max(speeds)
                    self.stats['avg_speed'] = avg_speed
                    self.stats['max_speed'] = max(self.stats['max_speed'], max_speed)
                    self.stats['trend_mode'] = 'speed'
                    self.stats['speed_records'].append(avg_speed)
                elif self.stats.get('speed_calibrated'):
                    self.stats['trend_mode'] = 'speed'
                    self.stats['speed_records'].append(0)
                else:
                    self.stats['trend_mode'] = 'motion_ai'
                    self.stats['speed_records'].append(float(motion_summary.get('max_motion_index', 0)))

                elapsed = time.perf_counter() - frame_start
                if elapsed > 0:
                    current_fps = 1 / elapsed
                    previous_fps = self.stats['processing_fps']
                    self.stats['processing_fps'] = current_fps if previous_fps <= 0 else previous_fps * 0.8 + current_fps * 0.2
                self.stats['processed_frames'] = frame_count

                self.display_frame(annotated_frame)
                self.update_stats()
                self.update_chart()
                self.check_danger(tracked_vehicles, height)

                time.sleep(1 / fps if fps > 0 else 0.033)

        except Exception as exc:
            failed = True
            message = f"❌ 检测线程异常：{exc}"
            self.update_warning(message, is_danger=True, event_key="detect_thread_error")
            self.run_on_ui(messagebox.showerror, "检测错误", message)
        finally:
            self.running = False
            if out is not None:
                out.release()
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            if not failed:
                if frame_count > 0:
                    video_text = os.path.basename(video_path) if video_path else "未保存"
                    save_info = f"✅ 已保存:\n📹 {video_text}\n📷 {len(self.stats['saved_screenshots'])}张截图"
                    self.run_on_ui(self.save_path_label.config, text=save_info, fg='#00ff88')
                    self.update_warning("✅ 检测完成！结果已保存")
                    self.run_on_ui(self.btn_report.config, state='normal')
                else:
                    self.update_warning("⚠️ 输入源没有读取到有效画面")

            self.reset_buttons()
    
    def get_class_color(self, class_name, speed=0):
        """获取类别颜色"""
        if class_name == 'person':
            return (0, 210, 255)
        elif class_name in ['stop sign', 'traffic light', 'fire hydrant']:
            return (255, 0, 255)
        elif speed > float(self.speed_config.get("speed_limit_kmh", 80)):
            return (0, 0, 255)
        elif speed > 50:
            return (0, 165, 255)
        else:
            return (0, 255, 0)
    
    def draw_results(self, frame, tracked_vehicles, frame_height):
        """绘制检测结果"""
        annotated = frame.copy()
        label_items = []
        demo_mode = getattr(self, 'demo_mode', 'both')

        if self.roi_config.get("enabled", True) and self.roi_config.get("show_overlay", True):
            roi_points = self.get_roi_points(annotated.shape[1], annotated.shape[0])
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [roi_points], (30, 80, 45))
            annotated = cv2.addWeighted(overlay, 0.16, annotated, 0.84, 0)
            cv2.polylines(annotated, [roi_points], True, (56, 189, 248), 2)

        show_pedestrian_layers = demo_mode in ["both", "pedestrian"]
        if show_pedestrian_layers and self.crossing_config.get("enabled", True) and self.crossing_config.get("show_zone", True):
            crossing_points = self.get_crossing_zone_points(annotated.shape[1], annotated.shape[0])
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [crossing_points], (90, 40, 120))
            annotated = cv2.addWeighted(overlay, 0.18, annotated, 0.82, 0)
            cv2.polylines(annotated, [crossing_points], True, (244, 114, 182), 2)

        if show_pedestrian_layers and self.pedestrian_safety_config.get("enabled", True) and self.pedestrian_safety_config.get("show_zone", True):
            safety_points = self.get_pedestrian_safety_zone_points(annotated.shape[1], annotated.shape[0])
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [safety_points], (30, 30, 140))
            annotated = cv2.addWeighted(overlay, 0.14, annotated, 0.86, 0)
            cv2.polylines(annotated, [safety_points], True, (248, 113, 113), 2)
        
        for vid, vdata in tracked_vehicles.items():
            if not self.is_confirmed_track(vdata):
                continue

            x1, y1, x2, y2 = map(int, vdata['bbox'])
            class_name = vdata['class']
            
            speed = float(vdata.get('speed', 0))
            distance = float(vdata.get('distance', 0))
            
            color = self.get_class_color(class_name, speed)
            risk_level = self.get_risk_level(class_name, distance, speed)
            is_vehicle = class_name in self.vehicle_classes
            is_person = class_name == 'person'
            in_focus = (
                demo_mode == 'both'
                or (demo_mode == 'vehicle' and (is_vehicle or class_name in self.barrier_classes))
                or (demo_mode == 'pedestrian' and (is_vehicle or is_person))
            )
            draw_color = color if in_focus else (90, 90, 90)
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), draw_color, 3 if in_focus else 1)
            
            if len(vdata['track']) > 2:
                points = np.array(vdata['track'][-10:], np.int32)
                cv2.polylines(annotated, [points], False, draw_color, 2 if in_focus else 1)
            
            label_prefix = "车辆" if is_vehicle else "行人" if is_person else "路障"
            label = f"{label_prefix} ID:{vid} {self.get_class_display_name(class_name)} {distance:.1f}m"
            if class_name in self.speed_objects:
                speed_status = vdata.get('speed_status', '')
                if self.stats.get('speed_calibrated'):
                    label += f" {speed:.1f}km/h"
                    if speed_status in ["静止", "低速"]:
                        label += f" {speed_status}"
                else:
                    motion_status = vdata.get('motion_status', 'AI分析中')
                    motion_index = int(vdata.get('ai_speed_index', 0))
                    ego_status = self.stats.get('ego_status', '初始化')
                    label += f" AI:{motion_status}"
                    if motion_status not in ["静止", "AI初始化", "未知"]:
                        label += f" 指数:{motion_index}"
                    if ego_status not in ["本车静止", "初始化"]:
                        label += " 已补偿"
            if risk_level != "低":
                label += f" 风险:{risk_level}"
            crossing_level = vdata.get('crossing_risk_level', '无')
            if class_name == 'person' and crossing_level in ['中', '高']:
                label += f" 横穿:{crossing_level}"
            safety_level = vdata.get('pedestrian_safety_level', '无')
            if class_name in self.vehicle_classes and safety_level in ['中', '高']:
                label += f" 避险:{safety_level} {vdata.get('pedestrian_safety_direction', '')}"

            display_level = safety_level if safety_level in ['中', '高'] else crossing_level if crossing_level in ['中', '高'] else risk_level
            label_items.append((x1, y1, label, (draw_color[2], draw_color[1], draw_color[0]), display_level, in_focus))

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(annotated_rgb)
        draw = ImageDraw.Draw(pil_image)

        for x1, y1, label, accent, risk_level, in_focus in label_items:
            bbox = draw.textbbox((0, 0), label, font=self.label_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            label_x = max(8, min(x1, pil_image.width - text_w - 20))
            label_y = max(8, y1 - text_h - 14)
            bg = (185, 28, 28) if risk_level == "高" else (180, 83, 9) if risk_level == "中" else (15, 23, 42) if in_focus else (31, 41, 55)
            draw.rounded_rectangle(
                (label_x, label_y, label_x + text_w + 14, label_y + text_h + 10),
                radius=5,
                fill=bg,
                outline=accent,
                width=2
            )
            draw.text((label_x + 7, label_y + 4), label, font=self.label_font, fill=(255, 255, 255))

        if demo_mode == "vehicle":
            panel_lines = [
                f"车辆感知  累计车辆 {self.stats['unique_vehicles']}  同帧车辆 {self.stats['total_vehicles']}  静止车辆 {self.stats['stationary_targets']}",
                f"本车状态 {self.stats['ego_status']}  AI速度指数 {self.stats['max_motion_index']}  FPS {self.stats['processing_fps']:.1f}"
            ]
        elif demo_mode == "pedestrian":
            panel_lines = [
                f"行人避险  累计行人 {self.stats['unique_pedestrians']}  危险来车 {self.stats['pedestrian_safety_risks']}",
                f"横穿风险 {self.stats['current_crossing_risks']}  高风险 {self.stats['high_pedestrian_safety_risks']}  FPS {self.stats['processing_fps']:.1f}"
            ]
        else:
            panel_lines = [
                f"综合展示  累计车辆 {self.stats['unique_vehicles']}  行人 {self.stats['unique_pedestrians']}  稳定目标 {self.stats['stable_targets']}",
                f"行人避险 {self.stats['pedestrian_safety_risks']}  横穿风险 {self.stats['current_crossing_risks']}  FPS {self.stats['processing_fps']:.1f}"
            ]
        panel_w = 700
        panel_h = 82
        draw.rounded_rectangle((14, 14, panel_w, panel_h), radius=8, fill=(15, 23, 42), outline=(56, 189, 248), width=2)
        for idx, line in enumerate(panel_lines):
            draw.text((28, 26 + idx * 28), line, font=self.panel_font, fill=(240, 249, 255))
        
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def check_danger(self, tracked_vehicles, frame_height):
        """检查危险情况"""
        vehicle_distance = float(self.risk_config.get("vehicle_distance_m", 15))
        pedestrian_distance = float(self.risk_config.get("pedestrian_distance_m", 20))
        speed_limit = float(self.speed_config.get("speed_limit_kmh", 80))

        for vid, vdata in tracked_vehicles.items():
            if not self.is_confirmed_track(vdata):
                continue

            class_name = vdata['class']
            distance = float(vdata.get('distance', 0))
            speed = float(vdata.get('speed', 0))
            risk_level = self.get_risk_level(class_name, distance, speed)
            
            if class_name in ['car', 'truck', 'bus']:
                pedestrian_safety_level = vdata.get('pedestrian_safety_level', '无')
                warn_levels = self.pedestrian_safety_config.get("warn_levels", ["中", "高"])
                if pedestrian_safety_level in warn_levels:
                    reasons = "、".join(vdata.get('pedestrian_safety_reasons', [])) or "危险来车"
                    direction = vdata.get('pedestrian_safety_direction', '来车')
                    advice = vdata.get('pedestrian_safety_advice', '注意避让')
                    if self.update_warning(
                        f"🛡️ [{pedestrian_safety_level}风险] 行人视角{direction}：{reasons}，建议：{advice}",
                        is_danger=True,
                        event_key=f"pedestrian_safety_{vid}_{pedestrian_safety_level}"
                    ):
                        self.stats['pedestrian_safety_alerts'] += 1

                if distance < vehicle_distance:
                    self.update_warning(
                        f"⚠️ [{risk_level}风险] 车辆ID:{vid} 距离过近 ({distance:.1f}m)！",
                        is_danger=True,
                        event_key=f"near_vehicle_{vid}_{risk_level}"
                    )
                if speed > speed_limit:
                    self.update_warning(
                        f"⚠️ [{risk_level}风险] 车辆ID:{vid} 超速 ({speed:.1f}km/h)！",
                        is_danger=True,
                        event_key=f"speed_{vid}_{risk_level}"
                    )
            
            if class_name == 'person':
                if distance < pedestrian_distance:
                    self.update_warning(
                        f"⚠️ [{risk_level}风险] 行人ID:{vid} 靠近车辆 ({distance:.1f}m)！",
                        is_danger=True,
                        event_key=f"near_person_{vid}_{risk_level}"
                    )

                crossing_level = vdata.get('crossing_risk_level', '无')
                warn_levels = self.crossing_config.get("warn_levels", ["中", "高"])
                if crossing_level in warn_levels:
                    reasons = "、".join(vdata.get('crossing_reasons', [])) or "横穿风险"
                    nearest_vehicle = vdata.get('crossing_nearest_vehicle')
                    vehicle_text = f"，关联车辆ID:{nearest_vehicle}" if nearest_vehicle is not None else ""
                    if self.update_warning(
                        f"🚶 [{crossing_level}风险] 行人ID:{vid} 横穿马路风险：{reasons}{vehicle_text}",
                        is_danger=True,
                        event_key=f"crossing_{vid}_{crossing_level}"
                    ):
                        self.stats['crossing_risk_count'] += 1
    
    def display_frame(self, frame):
        """显示帧"""
        self.current_frame = frame.copy()
        if threading.get_ident() != self.ui_thread_id:
            self.run_on_ui(self.display_frame, frame.copy())
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        
        self.window.update_idletasks()
        display_width = self.video_frame.winfo_width()
        display_height = self.video_frame.winfo_height()
        if display_width < 80 or display_height < 80:
            display_width, display_height = 1000, 562

        src_width, src_height = img.size
        scale = max(display_width / max(src_width, 1), display_height / max(src_height, 1))
        resized_width = max(int(src_width * scale), display_width)
        resized_height = max(int(src_height * scale), display_height)
        img = img.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

        left = max((resized_width - display_width) // 2, 0)
        top = max((resized_height - display_height) // 2, 0)
        img = img.crop((left, top, left + display_width, top + display_height))
        
        photo = ImageTk.PhotoImage(img)
        self.video_label.config(image=photo, text="")
        self.video_label.image = photo
    
    def update_stats(self):
        """更新统计"""
        if threading.get_ident() != self.ui_thread_id:
            self.run_on_ui(self.update_stats)
            return

        self.stat_labels['total_vehicles'].config(text=str(self.stats['total_vehicles']))
        self.stat_labels['unique_vehicles'].config(text=str(self.stats['unique_vehicles']))
        self.stat_labels['unique_pedestrians'].config(text=str(self.stats['unique_pedestrians']))
        self.stat_labels['unique_barriers'].config(text=str(self.stats['unique_barriers']))
        self.stat_labels['stable_targets'].config(text=str(self.stats['stable_targets']))
        self.stat_labels['filtered_detections'].config(text=str(self.stats['filtered_detections']))
        self.stat_labels['ego_status'].config(text=str(self.stats['ego_status']))
        self.stat_labels['max_motion_index'].config(text=str(self.stats['max_motion_index']))
        self.stat_labels['pedestrian_safety_risks'].config(text=str(self.stats['pedestrian_safety_risks']))
        self.stat_labels['current_crossing_risks'].config(text=str(self.stats['current_crossing_risks']))
        self.stat_labels['stationary_targets'].config(text=str(self.stats['stationary_targets']))
        self.stat_labels['danger_count'].config(text=str(self.stats['danger_count']))
        self.stat_labels['processing_fps'].config(text=f"{self.stats['processing_fps']:.1f}")
    
    def update_chart(self):
        """更新图表"""
        if threading.get_ident() != self.ui_thread_id:
            self.run_on_ui(self.update_chart)
            return

        records = list(self.stats['speed_records'])
        trend_mode = self.stats.get('trend_mode', 'motion_ai')
        is_speed_chart = trend_mode == 'speed'

        self.ax.set_ylabel('km/h' if is_speed_chart else 'AI index', color='#cbd5e1', fontsize=8)
        self.line.set_color('#38bdf8' if is_speed_chart else '#fb7185')

        if records:
            x_values = list(range(len(records)))
            self.line.set_data(x_values, records)
            self.ax.set_xlim(0, max(len(records) - 1, 20))
            max_y = max(records) * 1.25 if records else 100
            default_y = 30 if is_speed_chart else 100
            self.ax.set_ylim(0, max(max_y, default_y))
            self.chart_hint.set_visible(False)
        else:
            self.line.set_data([], [])
            self.ax.set_xlim(0, 20)
            self.ax.set_ylim(0, 30 if is_speed_chart else 100)
            self.chart_hint.set_text("Waiting for speed data" if is_speed_chart else "Waiting for AI motion data")
            self.chart_hint.set_visible(True)

        self.ax.grid(True, color='#334155', alpha=0.45, linewidth=0.8)
        self.fig.tight_layout(pad=1.1)
        self.canvas.draw_idle()
    
    def save_screenshot(self):
        """手动保存截图"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(
            self.output_dir, 
            f"手动截图_{timestamp}.jpg"
        )
        
        if hasattr(self, 'current_frame'):
            cv2.imwrite(screenshot_path, self.current_frame)
            self.stats['saved_screenshots'].append(screenshot_path)
            self.update_warning(f"📷 已保存截图: {os.path.basename(screenshot_path)}")
            
            save_info = f"✅ 已保存:\n📹 {os.path.basename(self.stats['saved_video']) if self.stats['saved_video'] else '未保存'}\n📷 {len(self.stats['saved_screenshots'])}张截图"
            self.save_path_label.config(text=save_info, fg='#00ff88')
    
    def stop_detection(self):
        """停止检测"""
        self.running = False
        self.update_warning("⏹️ 检测已停止")
        
        if self.cap:
            self.cap.release()
        
        self.reset_buttons()
        self.btn_report.config(state='normal')
    
    def reset_buttons(self):
        """重置按钮"""
        if threading.get_ident() != self.ui_thread_id:
            self.run_on_ui(self.reset_buttons)
            return

        self.btn_upload.config(state='normal')
        self.btn_camera.config(state='normal')
        self.btn_start.config(state='normal' if self.video_path and self.source_type == "video" else 'disabled')
        self.btn_stop.config(state='disabled')
        self.btn_screenshot.config(state='disabled')
    
    def generate_report(self):
        """生成检测报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = {
            '检测时间': timestamp,
            '输入源类型': '摄像头实时输入' if self.source_type == "camera" else '本地视频文件',
            '输入源': self.get_source_label(),
            '源视频': self.video_path if self.source_type == "video" else None,
            '输出目录': self.output_dir,
            '保存文件': {
                '检测视频': os.path.basename(self.stats['saved_video']) if self.stats['saved_video'] else None,
                '截图数量': len(self.stats['saved_screenshots']),
                '截图列表': [os.path.basename(s) for s in self.stats['saved_screenshots']]
            },
            '统计数据': {
                '最大同帧车辆数': self.stats['total_vehicles'],
                '最大同帧行人数': self.stats['total_pedestrians'],
                '最大同帧路障数': self.stats['total_barriers'],
                '累计跟踪车辆数': self.stats['unique_vehicles'],
                '累计跟踪行人数': self.stats['unique_pedestrians'],
                '累计跟踪路障数': self.stats['unique_barriers'],
                '危险预警次数': self.stats['danger_count'],
                '速度状态': '已标定' if self.stats.get('speed_calibrated') else '待标定',
                '本车主体状态': self.stats['ego_status'],
                '本车背景运动像素': round(self.stats['ego_motion_px'], 2),
                '本车状态置信度': round(self.stats['ego_confidence'], 3),
                'AI最高速度指数': self.stats['max_motion_index'],
                'AI判定运动目标数': self.stats['moving_targets'],
                'AI判定静止目标数': self.stats['stationary_targets'],
                '当前横穿风险行人数': self.stats['current_crossing_risks'],
                '高风险横穿行人数': self.stats['high_crossing_risks'],
                '累计横穿风险预警': self.stats['crossing_risk_count'],
                '当前行人避险风险车辆数': self.stats['pedestrian_safety_risks'],
                '高风险行人避险车辆数': self.stats['high_pedestrian_safety_risks'],
                '累计行人避险预警': self.stats['pedestrian_safety_alerts'],
                '最高可信车速': f"{self.stats['max_speed']:.1f} km/h" if self.stats.get('speed_calibrated') else '待标定',
                '平均车速': f"{self.stats['avg_speed']:.1f} km/h" if self.stats.get('speed_calibrated') else '待标定',
                '实时处理FPS': f"{self.stats['processing_fps']:.1f}",
                '处理帧数': self.stats['processed_frames'],
                '过滤异常速度次数': self.stats['invalid_speed_count'],
                '智能过滤检测框次数': self.stats['filtered_detections'],
                '稳定目标数': self.stats['stable_targets']
            },
            '危险事件记录': self.stats['danger_events'],
            '技术参数': {
                '检测置信度': self.confidence,
                '使用设备': 'GPU' if torch.cuda.is_available() else 'CPU',
                '模型': self.model_path,
                '摄像头编号': self.camera_index,
                '摄像头分辨率': f"{self.camera_width}x{self.camera_height}",
                '焦距参数': self.focal_length,
                '像素尺度': self.camera_config.get("meters_per_pixel"),
                '限速阈值': self.speed_config.get("speed_limit_kmh", 80),
                '最高可信速度阈值': self.speed_config.get("max_valid_speed_kmh", 160),
                '位移死区像素': self.speed_config.get("min_movement_px", 3.5),
                '静止速度阈值': self.speed_config.get("stationary_speed_kmh", 3.0),
                '速度是否要求标定': self.speed_config.get("require_calibration", True),
                '静止锁定帧数': self.speed_config.get("stationary_frames", 5),
                'AI运动分析': self.motion_config,
                '行人横穿风险配置': self.crossing_config,
                '行人视角避险配置': self.pedestrian_safety_config,
                '中心点平滑系数': self.config.get("tracking", {}).get("center_smoothing", 0.65),
                '目标确认帧数': self.min_confirmed_hits,
                'ROI过滤': self.roi_config.get("enabled", True),
                '最小检测框占比': self.detection_config.get("min_box_area_ratio", 0.00025),
                '事件冷却时间': self.risk_config.get("event_cooldown_sec", 3)
            }
        }
        
        report_filename = f"检测报告_{timestamp}.json"
        report_path = os.path.join(self.output_dir, report_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        txt_report = f"""
========================================
        路眼：智能交通检测系统 - 检测报告
========================================

检测时间: {timestamp}
输入源类型: {'摄像头实时输入' if self.source_type == "camera" else '本地视频文件'}
输入源: {self.get_source_label()}

【统计数据】
最大同帧车辆数: {self.stats['total_vehicles']}
最大同帧行人数: {self.stats['total_pedestrians']}
最大同帧路障数: {self.stats['total_barriers']}
累计跟踪车辆数: {self.stats['unique_vehicles']}
累计跟踪行人数: {self.stats['unique_pedestrians']}
累计跟踪路障数: {self.stats['unique_barriers']}
危险预警: {self.stats['danger_count']} 次
速度状态: {'已标定' if self.stats.get('speed_calibrated') else '待标定'}
本车主体状态: {self.stats['ego_status']}
本车背景运动像素: {self.stats['ego_motion_px']:.2f}
本车状态置信度: {self.stats['ego_confidence']:.3f}
AI最高速度指数: {self.stats['max_motion_index']}
AI判定运动目标数: {self.stats['moving_targets']}
AI判定静止目标数: {self.stats['stationary_targets']}
当前横穿风险行人数: {self.stats['current_crossing_risks']}
高风险横穿行人数: {self.stats['high_crossing_risks']}
累计横穿风险预警: {self.stats['crossing_risk_count']}
当前行人避险风险车辆数: {self.stats['pedestrian_safety_risks']}
高风险行人避险车辆数: {self.stats['high_pedestrian_safety_risks']}
累计行人避险预警: {self.stats['pedestrian_safety_alerts']}
最高可信车速: {f"{self.stats['max_speed']:.1f} km/h" if self.stats.get('speed_calibrated') else '待标定'}
平均车速: {f"{self.stats['avg_speed']:.1f} km/h" if self.stats.get('speed_calibrated') else '待标定'}
实时处理FPS: {self.stats['processing_fps']:.1f}
处理帧数: {self.stats['processed_frames']}
过滤异常速度: {self.stats['invalid_speed_count']} 次
智能过滤检测框: {self.stats['filtered_detections']} 次
稳定目标数: {self.stats['stable_targets']}

【保存文件】
检测视频: {os.path.basename(self.stats['saved_video']) if self.stats['saved_video'] else '无'}
截图数量: {len(self.stats['saved_screenshots'])} 张
报告文件: {report_filename}

【危险事件】
"""
        for event in self.stats['danger_events']:
            txt_report += f"  [{event['time']}] {event['message']}\n"
        
        txt_report += f"""
【技术参数】
检测置信度: {self.confidence}
使用设备: {'GPU' if torch.cuda.is_available() else 'CPU'}
模型: {self.model_path}
摄像头编号: {self.camera_index}
摄像头分辨率: {self.camera_width}x{self.camera_height}
焦距参数: {self.focal_length}
像素尺度: {self.camera_config.get("meters_per_pixel")}
限速阈值: {self.speed_config.get("speed_limit_kmh", 80)} km/h
最高可信速度阈值: {self.speed_config.get("max_valid_speed_kmh", 160)} km/h
位移死区像素: {self.speed_config.get("min_movement_px", 3.5)}
静止速度阈值: {self.speed_config.get("stationary_speed_kmh", 3.0)} km/h
速度是否要求标定: {self.speed_config.get("require_calibration", True)}
静止锁定帧数: {self.speed_config.get("stationary_frames", 5)}
AI运动分析: {self.motion_config}
行人横穿风险配置: {self.crossing_config}
行人视角避险配置: {self.pedestrian_safety_config}
中心点平滑系数: {self.config.get("tracking", {}).get("center_smoothing", 0.65)}
目标确认帧数: {self.min_confirmed_hits}
ROI过滤: {self.roi_config.get("enabled", True)}
最小检测框占比: {self.detection_config.get("min_box_area_ratio", 0.00025)}
事件冷却时间: {self.risk_config.get("event_cooldown_sec", 3)} 秒

========================================
"""
        
        txt_path = os.path.join(self.output_dir, f"检测报告_{timestamp}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(txt_report)
        
        self.stats['report'] = report_path
        self.update_warning(f"📄 报告已生成: {report_filename}")
        
        save_info = f"""✅ 完整保存:
📹 视频: {os.path.basename(self.stats['saved_video']) if self.stats['saved_video'] else '无'}
📷 截图: {len(self.stats['saved_screenshots'])}张
📄 报告: {report_filename}"""
        self.save_path_label.config(text=save_info, fg='#00ff88')
        
        messagebox.showinfo("报告生成", 
            f"检测报告已保存至:\n{self.output_dir}\n\n"
            f"包含:\n"
            f"• 检测视频\n"
            f"• {len(self.stats['saved_screenshots'])}张截图\n"
            f"• JSON报告\n"
            f"• TXT报告")
    
    def run(self):
        """运行"""
        self.window.mainloop()


if __name__ == "__main__":
    app = SmartTrafficSystem()
    app.run()
