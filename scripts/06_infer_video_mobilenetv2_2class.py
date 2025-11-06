#!/usr/bin/env python3
"""
Video inference using MobileNetV2 for face mask detection (2 classes)
Uses YOLOv8-face for face detection and MobileNetV2 for mask classification
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import torch
import torch.nn as nn
import numpy as np
import argparse
from ultralytics import YOLO
from torchvision import transforms
import time

# Import models with fallback
try:
    from torchvision import models
except ImportError:
    print("Warning: torchvision.models not available, using alternative approach")
    models = None

from utils.paths import ensure_dir


def create_model(num_classes=2, pretrained=True):
    """Create MobileNetV2 model for 2 classes"""
    if models is None:
        raise RuntimeError("torchvision.models not available. Please install torchvision properly.")
    
    if pretrained:
        try:
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        except:
            # Fallback for older torchvision versions
            model = models.mobilenet_v2(pretrained=True)
    else:
        model = models.mobilenet_v2(pretrained=False)
    
    # Replace classifier for 2 classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def preprocess_image(image, target_size=(500, 500)):
    """Preprocess image for MobileNetV2"""
    # Resize image
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (rgb.astype(np.float32) / 255.0 - mean) / std
    
    # Convert to tensor
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float()
    
    return tensor.unsqueeze(0)  # Add batch dimension


def detect_faces(image, face_model, conf_threshold=0.5):
    """Detect faces using YOLOv8-face"""
    try:
        results = face_model(image, conf=conf_threshold, verbose=False)
        faces = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    
                    faces.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf
                    })
        
        return faces
    except Exception as e:
        print(f"Error in face detection: {e}")
        return []


def crop_face(image, bbox, padding=0.1):
    """Crop face from image with padding"""
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    
    # Add padding
    pad_w = int((x2 - x1) * padding)
    pad_h = int((y2 - y1) * padding)
    
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)
    
    return image[y1:y2, x1:x2]


def predict_mask(image, model, device, class_names):
    """Predict mask class for a single image"""
    try:
        # Preprocess image
        input_tensor = preprocess_image(image)
        input_tensor = input_tensor.to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        # Get results
        class_id = predicted.item()
        confidence = float(torch.max(probs).item())
        class_name = class_names[class_id]
        
        return class_name, confidence, probs[0].cpu().numpy()
        
    except Exception as e:
        print(f"Error predicting mask: {e}")
        return "unknown", 0.0, np.array([0.5, 0.5])


def draw_box_with_label(image, bbox, label, class_id, confidence):
    """Draw bounding box with label on image"""
    x1, y1, x2, y2 = bbox
    
    # Colors for 2 classes
    colors = {
        0: (0, 255, 0),    # no_mask - Green
        1: (0, 0, 255)     # mask - Red
    }
    
    color = colors.get(class_id, (255, 255, 255))
    
    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Draw label background
    label_text = f"{label}: {confidence:.2f}"
    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
    
    # Draw label text
    cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image


def main():
    parser = argparse.ArgumentParser(description='Video inference with MobileNetV2 (2 classes)')
    parser.add_argument('--video_path', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--model_path', type=str, default='models3/image_baseline/mobilenetv2_best_2class.pt',
                       help='Path to trained MobileNetV2 model')
    parser.add_argument('--face_weights', type=str, default='weights/yolov8n-face.pt',
                       help='Path to YOLOv8-face weights')
    parser.add_argument('--output_path', type=str, default='Results3/video_output_2class.mp4',
                       help='Path to output video file')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                       help='Confidence threshold for face detection')
    parser.add_argument('--show_video', action='store_true',
                       help='Show video during processing')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--img_size', type=int, default=500,
                       help='Input image size for MobileNetV2')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Video Inference with MobileNetV2 (2 Classes)")
    print("="*60)
    print(f"Video path: {args.video_path}")
    print(f"Model path: {args.model_path}")
    print(f"Face weights: {args.face_weights}")
    print(f"Output path: {args.output_path}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print(f"Show video: {args.show_video}")
    print(f"Device: {args.device}")
    print(f"Image size: {args.img_size}")
    print(f"Classes: no_mask, mask")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU instead")
    
    print(f"Using device: {device}")
    
    # Load face detection model
    print(f"\nLoading face detection model: {args.face_weights}")
    face_model = YOLO(args.face_weights)
    
    # Load MobileNetV2 model
    print(f"Loading MobileNetV2 model: {args.model_path}")
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    num_classes = 2
    model = create_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()
    
    class_names = ['no_mask', 'mask']
    print(f"Model loaded with classes: {class_names}")
    
    # Open video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {args.video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))
    
    # Create output directory
    ensure_dir(os.path.dirname(args.output_path))
    
    print(f"\nStarting video processing...")
    print("Press 'q' to quit, 'p' to pause/resume")
    
    frame_count = 0
    start_time = time.time()
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect faces
            faces = detect_faces(frame, face_model, args.conf_threshold)
            
            # Process each face
            for face in faces:
                bbox = face['bbox']
                face_conf = face['confidence']
                
                # Crop face
                face_crop = crop_face(frame, bbox)
                
                if face_crop.size > 0:
                    # Predict mask
                    class_name, mask_conf, probs = predict_mask(face_crop, model, device, class_names)
                    
                    # Draw results
                    class_id = class_names.index(class_name) if class_name in class_names else 0
                    frame = draw_box_with_label(frame, bbox, class_name, class_id, mask_conf)
            
            # Write frame
            out.write(frame)
            
            # Show progress
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - FPS: {fps_current:.1f}")
        
        # Show video if requested
        if args.show_video:
            cv2.imshow('Face Mask Detection (2 Classes)', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
        else:
            # Check for quit without showing video
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    # Cleanup
    cap.release()
    out.release()
    if args.show_video:
        cv2.destroyAllWindows()
    
    # Print summary
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed
    
    print(f"\n" + "="*60)
    print("Video Processing Complete!")
    print("="*60)
    print(f"Processed frames: {frame_count}")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Output saved: {args.output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
