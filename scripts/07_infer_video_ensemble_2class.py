#!/usr/bin/env python3
"""
Video ensemble inference (2 classes: no_mask, mask)
- Detection: YOLOv8-face
- Tracking: Lightweight ByteTrack-style (greedy IoU matching, max_age, min_hits)
- Classification: MobileNetV2 (image) + RandomForest (landmarks via MediaPipe)
- Fusion: Weighted average of probabilities, EMA smoothing + hysteresis/debounce
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision import transforms

try:
    from torchvision import models
except ImportError:
    models = None
    print("Warning: torchvision.models not available")

try:
    import joblib
except ImportError:
    joblib = None
    print("Warning: joblib not available; cannot load RF model")

try:
    import mediapipe as mp
    MP_AVAILABLE = True
    mp_face_mesh = mp.solutions.face_mesh
except ImportError:
    MP_AVAILABLE = False
    print("Warning: mediapipe not available; landmark path will be disabled")

from utils.paths import ensure_dir

# Global container for RF feature column ordering (loaded from metadata if available)
RF_FEATURE_COLUMNS = None


# ------------------------------- Utils ---------------------------------

def iou(b1, b2):
    """Compute IoU between two boxes (x1,y1,x2,y2)."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    inter = w * h
    area1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    area2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def greedy_match(detections, tracks, iou_threshold=0.3):
    """Greedy IoU matching between detections and existing tracks.
    Returns (matches, unmatched_det_indices, unmatched_track_indices)
    """
    if len(detections) == 0 or len(tracks) == 0:
        return [], list(range(len(detections))), list(range(len(tracks)))

    ious = np.zeros((len(detections), len(tracks)), dtype=np.float32)
    for i, d in enumerate(detections):
        for j, t in enumerate(tracks):
            ious[i, j] = iou(d['bbox'], t.bbox)

    matches = []
    used_dets = set()
    used_tracks = set()
    # Greedy by highest IoU first
    pairs = [(i, j, ious[i, j]) for i in range(len(detections)) for j in range(len(tracks))]
    pairs.sort(key=lambda x: x[2], reverse=True)
    for i, j, v in pairs:
        if v < iou_threshold:
            break
        if i in used_dets or j in used_tracks:
            continue
        matches.append((i, j))
        used_dets.add(i)
        used_tracks.add(j)

    unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
    unmatched_tracks = [j for j in range(len(tracks)) if j not in used_tracks]
    return matches, unmatched_dets, unmatched_tracks


# --------------------------- Landmark features --------------------------

def extract_landmark_features(image_bgr):
    """Extract lightweight landmark features from a face crop using MediaPipe.
    Returns dict of features or None when failed.
    """
    if not MP_AVAILABLE:
        return None
    try:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            res = face_mesh.process(image_rgb)
            if not res.multi_face_landmarks:
                return None
            lm = res.multi_face_landmarks[0]
        h, w = image_bgr.shape[:2]

        def L(idx):
            p = lm.landmark[idx]
            return int(p.x * w), int(p.y * h), float(p.visibility)

        # Minimal set similar to scripts3/02_extract_landmarks_yolo_2class.py
        def dist(a, b):
            return math.hypot(a[0]-b[0], a[1]-b[1])

        # Key points
        left_eye_left = L(33)
        left_eye_right = L(133)
        left_eye_top = L(159)
        left_eye_bottom = L(145)
        right_eye_left = L(362)
        right_eye_right = L(263)
        right_eye_top = L(386)
        right_eye_bottom = L(374)
        nose_tip = L(1)
        nose_bridge = L(6)
        mouth_left = L(61)
        mouth_right = L(291)
        mouth_top = L(13)
        mouth_bottom = L(14)
        face_left = L(172)
        face_right = L(397)
        face_top = L(10)
        face_bottom = L(152)

        left_eye_w = abs(left_eye_right[0]-left_eye_left[0])
        left_eye_h = abs(left_eye_bottom[1]-left_eye_top[1])
        right_eye_w = abs(right_eye_right[0]-right_eye_left[0])
        right_eye_h = abs(right_eye_bottom[1]-right_eye_top[1])

        nose_w = abs(L(35)[0]-L(31)[0])
        nose_h = abs(nose_tip[1]-nose_bridge[1])

        mouth_w = abs(mouth_right[0]-mouth_left[0])
        mouth_h = abs(mouth_bottom[1]-mouth_top[1])

        face_w = abs(face_right[0]-face_left[0])
        face_h = abs(face_bottom[1]-face_top[1])
        face_area = max(1, face_w * face_h)

        # Eye/mouth centers for eye-mouth distance
        left_eye_center = ((left_eye_left[0] + left_eye_right[0]) // 2,
                           (left_eye_top[1] + left_eye_bottom[1]) // 2)
        right_eye_center = ((right_eye_left[0] + right_eye_right[0]) // 2,
                            (right_eye_top[1] + right_eye_bottom[1]) // 2)
        eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                      (left_eye_center[1] + right_eye_center[1]) // 2)
        mouth_center = ((mouth_left[0] + mouth_right[0]) // 2,
                        (mouth_top[1] + mouth_bottom[1]) // 2)
        eye_mouth_distance = abs(eye_center[1] - mouth_center[1])

        # Average mouth visibility over a subset of landmarks around the lips
        mouth_idxs = [13, 14, 61, 62, 63, 64, 65, 66, 67, 291, 308, 324, 318, 402, 317, 292]
        mouth_vis_list = []
        for idx in mouth_idxs:
            _, _, vis = L(idx)
            mouth_vis_list.append(vis)
        mouth_visibility_avg = float(np.mean(mouth_vis_list)) if mouth_vis_list else float(mouth_top[2])

        features = {
            'left_eye_width': left_eye_w / math.sqrt(face_area),
            'left_eye_height': left_eye_h / math.sqrt(face_area),
            'right_eye_width': right_eye_w / math.sqrt(face_area),
            'right_eye_height': right_eye_h / math.sqrt(face_area),
            'left_eye_aspect_ratio': left_eye_h / max(1, left_eye_w),
            'right_eye_aspect_ratio': right_eye_h / max(1, right_eye_w),
            'nose_width': nose_w / math.sqrt(face_area),
            'nose_height': nose_h / math.sqrt(face_area),
            'nose_aspect_ratio': nose_h / max(1, nose_w),
            'mouth_width': mouth_w / math.sqrt(face_area),
            'mouth_height': mouth_h / math.sqrt(face_area),
            'mouth_aspect_ratio': mouth_h / max(1, mouth_w),
            'face_width': face_w / math.sqrt(face_area),
            'face_height': face_h / math.sqrt(face_area),
            'face_aspect_ratio': face_h / max(1, face_w),
            'mouth_visibility': float(mouth_top[2]),
            'nose_visibility': float(nose_tip[2]),
            'chin_visibility': float(face_bottom[2]),
            'eye_mouth_distance': eye_mouth_distance / math.sqrt(face_area),
            'mouth_visibility_avg': mouth_visibility_avg
        }
        return features
    except Exception as e:
        print(f"Landmark error: {e}")
        return None


# ------------------------------- Models --------------------------------

def create_mobilenetv2(num_classes=2, pretrained=True):
    if models is None:
        raise RuntimeError("torchvision.models not available")
    if pretrained:
        try:
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        except Exception:
            model = models.mobilenet_v2(pretrained=True)
    else:
        model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def preprocess_image(image_bgr, target_size=(500, 500)):
    resized = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_LANCZOS4)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (rgb.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor


def crop_with_padding(frame, bbox, padding=0.1):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    pw = int((x2 - x1) * padding)
    ph = int((y2 - y1) * padding)
    x1 = max(0, x1 - pw)
    y1 = max(0, y1 - ph)
    x2 = min(w, x2 + pw)
    y2 = min(h, y2 + ph)
    crop = frame[y1:y2, x1:x2]
    return crop


# ------------------------------- Tracker --------------------------------
class Track:
    def __init__(self, track_id, bbox, cls_names, ema_beta=0.85):
        self.id = track_id
        self.bbox = bbox
        self.last_seen = 0
        self.hits = 1
        self.age = 0
        self.cls_names = cls_names
        self.ema_beta = ema_beta
        self.ema_probs = np.array([0.5, 0.5], dtype=np.float32)
        self.window_preds = []  # keep last N hard preds
        self.label = None
        self.switch_counter = 0

    def update_bbox(self, bbox):
        self.bbox = bbox
        self.hits += 1
        self.age = 0

    def decay(self):
        self.age += 1

    def update_probs(self, probs, min_conf_for_update=0.55):
        m = float(np.max(probs))
        if m < min_conf_for_update:
            return
        self.ema_probs = self.ema_beta * self.ema_probs + (1 - self.ema_beta) * probs

    def majority_vote(self, N=9):
        if not self.window_preds:
            return None
        vals, counts = np.unique(self.window_preds[-N:], return_counts=True)
        return int(vals[np.argmax(counts)])

    def decide_label(self, t_on=0.7, t_off=0.6, margin=0.1, K=3, window_N=9):
        # Hysteresis on EMA probs
        cid = int(np.argmax(self.ema_probs))
        conf = float(np.max(self.ema_probs))
        other = 1 - cid
        gap = self.ema_probs[cid] - self.ema_probs[other]

        if self.label is None:
            if conf >= t_on:
                self.label = cid
                self.switch_counter = 0
            return self.label, conf

        if cid != self.label and conf >= t_on and gap >= margin:
            self.switch_counter += 1
            if self.switch_counter >= K:
                self.label = cid
                self.switch_counter = 0
        else:
            # no confident switch
            if conf <= t_off:
                # try majority window if EMA uncertain
                mv = self.majority_vote(window_N)
                if mv is not None:
                    self.label = mv
            self.switch_counter = 0

        return self.label, conf


class ByteTrackLite:
    def __init__(self, cls_names, iou_threshold=0.3, max_age=30, min_hits=2, ema_beta=0.85,
                 t_on=0.7, t_off=0.6, margin=0.1, K=3, window_N=9):
        self.cls_names = cls_names
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.ema_beta = ema_beta
        self.t_on = t_on
        self.t_off = t_off
        self.margin = margin
        self.K = K
        self.window_N = window_N
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        # detections: list of {'bbox':(x1,y1,x2,y2), 'conf':float, 'probs': np.array([p0,p1])}
        # Age existing tracks
        for t in self.tracks:
            t.decay()

        matches, unmatched_det, unmatched_trk = greedy_match(detections, self.tracks, self.iou_threshold)

        # Update matched tracks
        for di, tj in matches:
            d = detections[di]
            t = self.tracks[tj]
            t.update_bbox(d['bbox'])
            if 'probs' in d and d['probs'] is not None:
                t.update_probs(d['probs'])
                pred = int(np.argmax(d['probs']))
                t.window_preds.append(pred)

        # Create new tracks for unmatched detections
        for di in unmatched_det:
            d = detections[di]
            t = Track(self.next_id, d['bbox'], self.cls_names, ema_beta=self.ema_beta)
            self.next_id += 1
            if 'probs' in d and d['probs'] is not None:
                t.update_probs(d['probs'])
                pred = int(np.argmax(d['probs']))
                t.window_preds.append(pred)
            self.tracks.append(t)

        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]

        # Decide labels
        outputs = []
        for t in self.tracks:
            label, conf = t.decide_label(self.t_on, self.t_off, self.margin, self.K, self.window_N)
            outputs.append({
                'id': t.id,
                'bbox': t.bbox,
                'label': label,
                'label_name': self.cls_names[label] if label is not None else 'unknown',
                'conf': conf,
                'probs_ema': t.ema_probs.copy()
            })
        return outputs


# ------------------------------- Inference ------------------------------

def detect_faces(frame, face_model, conf_threshold=0.5):
    try:
        results = face_model(frame, conf=conf_threshold, verbose=False)
        faces = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for b in boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                conf = float(b.conf[0].cpu().numpy())
                faces.append({'bbox': (x1, y1, x2, y2), 'conf': conf})
        return faces
    except Exception as e:
        print(f"Face detection error: {e}")
        return []


def predict_mobilenet(face_crop, model, device):
    input_tensor = preprocess_image(face_crop).to(device)
    with torch.no_grad():
        out = model(input_tensor)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    return probs


def predict_landmark_rf(face_crop, rf_model, rf_scaler):
    if not (MP_AVAILABLE and rf_model is not None and rf_scaler is not None):
        return None
    features = extract_landmark_features(face_crop)
    if features is None:
        return None
    # Build feature vector aligned to training columns if available
    n_expected = getattr(rf_scaler, 'n_features_in_', None)
    cols = None
    global RF_FEATURE_COLUMNS
    if RF_FEATURE_COLUMNS:
        cols = RF_FEATURE_COLUMNS
    elif hasattr(rf_scaler, 'feature_names_in_'):
        cols = list(rf_scaler.feature_names_in_)
    else:
        # Fallback to sorted keys (last resort)
        cols = sorted(features.keys())

    x = np.array([features.get(c, 0.0) for c in cols], dtype=np.float32).reshape(1, -1)
    # Adjust shape to match scaler expectation if necessary
    if n_expected is not None and x.shape[1] != n_expected:
        if x.shape[1] > n_expected:
            x = x[:, :n_expected]
        else:
            pad = np.zeros((1, n_expected - x.shape[1]), dtype=np.float32)
            x = np.concatenate([x, pad], axis=1)
    X_scaled = rf_scaler.transform(x)
    probs = rf_model.predict_proba(X_scaled)[0]
    return probs


def draw_box_with_label(image, bbox, track_id, label_name, conf):
    x1, y1, x2, y2 = bbox
    # Map label to background color; avoid white as default
    # Color scheme requested by user:
    # mask = green, no_mask = red, unknown = orange
    colors = {
        'mask': (0, 255, 0),        # green
        'no_mask': (0, 0, 255),     # red
        'unknown': (0, 165, 255),   # orange for pending/unknown
        'pending': (0, 165, 255)
    }
    bg_color = colors.get(label_name if label_name is not None else 'unknown', (0, 165, 255))

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), bg_color, 2)

    # Prepare text and compute contrasting text color
    safe_label = label_name if label_name is not None else 'pending'
    text = f"ID {track_id} | {safe_label} {conf:.2f}"
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    # Luminance to choose white/black text for contrast
    luminance = 0.2126 * bg_color[2] + 0.7152 * bg_color[1] + 0.0722 * bg_color[0]
    text_color = (0, 0, 0) if luminance > 180 else (255, 255, 255)

    # Draw filled label background and text
    cv2.rectangle(image, (x1, max(0, y1 - text_h - 10)), (x1 + text_w, y1), bg_color, -1)
    cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    return image


def main():
    parser = argparse.ArgumentParser(description='Video Ensemble Inference (2 classes) with ByteTrack-style tracking')
    parser.add_argument('--video_path', type=str, required=True, help='Input video path')
    parser.add_argument('--output_path', type=str, default='Results3/video_output_ensemble_2class.mp4', help='Output video path')
    parser.add_argument('--image_model_path', type=str, default='models3/image_baseline/mobilenetv2_best_2class.pt', help='MobileNetV2 weights path')
    parser.add_argument('--landmark_model_path', type=str, default='models3/landmark_ml/best_landmark_rf_2class.pkl', help='RandomForest model path')
    parser.add_argument('--landmark_scaler_path', type=str, default='models3/landmark_ml/scaler_rf_2class.pkl', help='Scaler path for RF')
    parser.add_argument('--landmark_metadata_path', type=str, default='models3/landmark_ml/metadata_rf_2class.pkl', help='Metadata path to load feature column order')
    parser.add_argument('--face_weights', type=str, default='weights/yolov8n-face.pt', help='YOLOv8-face weights')
    parser.add_argument('--alpha', type=float, default=0.7, help='Weight for MobileNetV2 in fusion')
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='Face detection confidence threshold')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--show_video', action='store_true', help='Show live video')
    parser.add_argument('--save_csv', action='store_true', help='Save per-frame results CSV')
    parser.add_argument('--iou_threshold', type=float, default=0.3, help='IoU threshold for matching')
    parser.add_argument('--max_age', type=int, default=30, help='Max age to keep lost tracks')
    parser.add_argument('--ema_beta', type=float, default=0.85, help='EMA beta for probs smoothing')
    parser.add_argument('--t_on', type=float, default=0.7, help='Hysteresis ON threshold')
    parser.add_argument('--t_off', type=float, default=0.6, help='Hysteresis OFF threshold')
    parser.add_argument('--margin', type=float, default=0.1, help='Margin for class switch')
    parser.add_argument('--K', type=int, default=3, help='Consecutive frames to confirm switch')
    parser.add_argument('--window_N', type=int, default=9, help='Majority window size')
    args = parser.parse_args()

    print("="*60)
    print("Ensemble Inference (2 Classes) with ByteTrack-style Tracking")
    print("="*60)
    print(f"Video: {args.video_path}")
    print(f"Output: {args.output_path}")
    print(f"Alpha (image weight): {args.alpha}")
    print(f"Device: {args.device}")
    print(f"YOLO Face: {args.face_weights}")
    print(f"Landmark RF: {args.landmark_model_path}")

    classes = ['no_mask', 'mask']

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")

    # Load detectors and classifiers
    face_model = YOLO(args.face_weights)

    img_model = create_mobilenetv2(num_classes=2, pretrained=False)
    if not os.path.exists(args.image_model_path):
        print(f"Error: image model not found: {args.image_model_path}")
        return
    state = torch.load(args.image_model_path, map_location=device, weights_only=False)
    img_model.load_state_dict(state)
    img_model = img_model.to(device).eval()

    rf_model = None
    rf_scaler = None
    if joblib is not None and os.path.exists(args.landmark_model_path) and os.path.exists(args.landmark_scaler_path):
        try:
            rf_model = joblib.load(args.landmark_model_path)
            rf_scaler = joblib.load(args.landmark_scaler_path)
            print("✓ Landmark RF model loaded")
            # Try load metadata for feature column ordering
            if os.path.exists(args.landmark_metadata_path):
                try:
                    meta = joblib.load(args.landmark_metadata_path)
                    cols = meta.get('feature_columns') if isinstance(meta, dict) else None
                    if cols:
                        global RF_FEATURE_COLUMNS
                        RF_FEATURE_COLUMNS = list(cols)
                        print(f"✓ Loaded RF feature columns: {len(RF_FEATURE_COLUMNS)}")
                except Exception as e:
                    print(f"Warning: cannot load metadata columns: {e}")
        except Exception as e:
            print(f"Warning: failed to load RF/scaler: {e}")
    else:
        print("⚠️  Landmark path unavailable; ensemble will use image model only when needed")

    # Tracker
    tracker = ByteTrackLite(
        classes,
        iou_threshold=args.iou_threshold,
        max_age=args.max_age,
        ema_beta=args.ema_beta,
        t_on=args.t_on,
        t_off=args.t_off,
        margin=args.margin,
        K=args.K,
        window_N=args.window_N
    )

    # Video IO
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video: {args.video_path}")
        return
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ensure_dir(os.path.dirname(args.output_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output_path, fourcc, fps, (w, h))

    print(f"Video: {w}x{h} @ {fps} FPS, frames: {total}")

    frame_idx = 0
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Detect faces
        dets = detect_faces(frame, face_model, args.conf_threshold)

        # Classify each detection and build detection entries with probs
        detection_entries = []
        for d in dets:
            bbox = d['bbox']
            crop = crop_with_padding(frame, bbox)
            if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                continue
            p_img = predict_mobilenet(crop, img_model, device)
            p_land = predict_landmark_rf(crop, rf_model, rf_scaler)
            if p_land is None:
                p_final = p_img
            else:
                p_final = args.alpha * p_img + (1.0 - args.alpha) * p_land
            p_final = p_final / max(1e-8, p_final.sum())
            detection_entries.append({'bbox': bbox, 'conf': d['conf'], 'probs': p_final})

        # Update tracker
        tracked = tracker.update(detection_entries)

        # Draw
        for t in tracked:
            label_name = t['label_name']
            conf = float(np.max(t['probs_ema']))
            frame = draw_box_with_label(frame, t['bbox'], t['id'], label_name, conf)

        writer.write(frame)

        if frame_idx % max(1, fps) == 0:
            elapsed = time.time() - start
            cur_fps = frame_idx / max(1e-6, elapsed)
            print(f"Frame {frame_idx}/{total} | FPS: {cur_fps:.1f} | Active tracks: {len(tracker.tracks)}")

        if args.show_video:
            cv2.imshow('Ensemble (2-class) + ByteTrack', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    writer.release()
    if args.show_video:
        cv2.destroyAllWindows()

    elapsed = time.time() - start
    print("="*60)
    print("Ensemble Inference Complete")
    print("="*60)
    print(f"Frames: {frame_idx}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Average FPS: {frame_idx / max(1e-6, elapsed):.2f}")
    print(f"Output saved: {args.output_path}")


if __name__ == '__main__':
    main()
