#!/usr/bin/env python3
"""
Detect and crop faces using YOLOv8-face, selecting the clearest face
Uses YOLO format labels for 2-class classification (no_mask vs mask)
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from ultralytics import YOLO

from utils.paths import ensure_dir


def load_yolo_labels(data_dir, split):
    """Load YOLO format labels and convert to 2 classes"""
    images_dir = os.path.join(data_dir, split, 'images')
    labels_dir = os.path.join(data_dir, split, 'labels')
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"Error: YOLO directories not found for {split}")
        return []
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    labels_data = []
    for img_file in image_files:
        # Get corresponding label file
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        if not os.path.exists(label_path):
            continue
            
        # Read label file
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) != 5:
                continue
                
            class_id = int(parts[0])
            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Map class_id to 2-class system
            if class_id == 0:
                class_name = 'no_mask'      # without_mask
            elif class_id == 1:
                class_name = 'mask'         # mask_incorrect → mask
            elif class_id == 2:
                class_name = 'mask'         # with_mask → mask
            else:
                print(f"Warning: Unknown class_id {class_id}, skipping...")
                continue
            
            labels_data.append({
                'image_file': img_file,
                'image_path': os.path.join(images_dir, img_file),
                'class_id': class_id,
                'class_name': class_name,
                'center_x': center_x,
                'center_y': center_y,
                'width': width,
                'height': height
            })
    
    print(f"Loaded {len(labels_data)} face annotations from {split}")
    return labels_data


def detect_faces_in_image(image_path, model, conf_threshold=0.5):
    """Detect faces in a single image using YOLOv8-face"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        # Run face detection
        results = model(image, conf=conf_threshold, verbose=False)
        
        faces = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    
                    # Calculate face area (for selecting clearest face)
                    face_area = (x2 - x1) * (y2 - y1)
                    
                    faces.append({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'confidence': conf,
                        'area': face_area
                    })
        
        return faces
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return []


def select_clearest_face(faces):
    """Select the clearest face (highest confidence * area)"""
    if not faces:
        return None
    
    # Score = confidence * area (normalized)
    max_area = max(face['area'] for face in faces)
    for face in faces:
        face['score'] = face['confidence'] * (face['area'] / max_area)
    
    # Return face with highest score
    return max(faces, key=lambda x: x['score'])


def crop_face_from_detection(image, face_box, target_size=(500, 500), padding=0.1):
    """Crop face from image using detection box and resize"""
    try:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = face_box['x1'], face_box['y1'], face_box['x2'], face_box['y2']
        
        # Add padding
        pad_w = int((x2 - x1) * padding)
        pad_h = int((y2 - y1) * padding)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        # Crop face
        face_crop = image[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            return None
        
        # Resize to target size
        resized_crop = cv2.resize(face_crop, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        return resized_crop, (x1, y1, x2, y2)
        
    except Exception as e:
        print(f"Error cropping face: {e}")
        return None, None


def process_dataset_split(data_dir, split, model, output_dir, target_size=(500, 500), conf_threshold=0.5):
    """Process a single dataset split"""
    print(f"\nProcessing {split} set...")
    print(f"Target resize size: {target_size[0]}x{target_size[1]}")
    print(f"Confidence threshold: {conf_threshold}")
    
    # Load YOLO labels
    labels_data = load_yolo_labels(data_dir, split)
    
    if not labels_data:
        print(f"Error: No YOLO labels found for {split}")
        return []
    
    crops_data = []
    crops_dir = os.path.join(output_dir, 'crops', split)
    ensure_dir(crops_dir)
    
    # Create class directories for 2 classes
    for class_name in ['no_mask', 'mask']:
        class_crops_dir = os.path.join(crops_dir, class_name)
        ensure_dir(class_crops_dir)
    
    # Process each image
    for idx, bbox_data in enumerate(tqdm(labels_data, desc=f"Processing {split}")):
        image_path = bbox_data['image_path']
        class_name = bbox_data['class_name']
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image: {image_path}")
            continue
        
        # Detect faces
        faces = detect_faces_in_image(image_path, model, conf_threshold)
        
        if not faces:
            print(f"Warning: No faces detected in {image_path}")
            continue
        
        # Select clearest face
        clearest_face = select_clearest_face(faces)
        if clearest_face is None:
            print(f"Warning: Could not select clearest face in {image_path}")
            continue
        
        # Crop face
        face_crop, crop_coords = crop_face_from_detection(image, clearest_face, target_size)
        
        if face_crop is None:
            print(f"Warning: Could not crop face from {image_path}")
            continue
        
        # Create output filename
        base_name = os.path.splitext(bbox_data['image_file'])[0]
        crop_filename = f"{base_name}_face_0.jpg"
        crop_path = os.path.join(crops_dir, class_name, crop_filename)
        
        # Save cropped face
        success = cv2.imwrite(crop_path, face_crop)
        
        if success:
            crops_data.append({
                'image_path': image_path,
                'crop_path': crop_path,
                'class_name': class_name,
                'face_id': 0,
                'x1': crop_coords[0],
                'y1': crop_coords[1],
                'x2': crop_coords[2],
                'y2': crop_coords[3],
                'confidence': clearest_face['confidence'],
                'area': clearest_face['area'],
                'score': clearest_face['score'],
                'original_size': (image.shape[1], image.shape[0]),
                'resized_size': target_size
            })
        else:
            print(f"Warning: Could not save crop: {crop_path}")
    
    return crops_data


def main():
    parser = argparse.ArgumentParser(description='Detect and crop faces using YOLOv8-face with 2-class YOLO labels')
    parser.add_argument('--data_dir', type=str, default='Datasets2',
                       help='Path to YOLO format dataset directory')
    parser.add_argument('--output_dir', type=str, default='data3/processed',
                       help='Output directory for processed data')
    parser.add_argument('--face_weights', type=str, default='weights/yolov8n-face.pt',
                       help='Path to YOLOv8-face weights')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                       help='Confidence threshold for face detection')
    parser.add_argument('--target_size', type=int, nargs=2, default=[500, 500],
                       help='Target size for resizing images (width height)')
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'],
                       help='Dataset splits to process')
    
    args = parser.parse_args()
    target_size = tuple(args.target_size)
    
    print("="*60)
    print("Face Detection and Cropping with YOLOv8-face (2 Classes)")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Face detection model: {args.face_weights}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print(f"Target resize size: {target_size[0]}x{target_size[1]}")
    print(f"Splits to process: {args.splits}")
    print(f"Classes: no_mask, mask")
    
    # Load face detection model
    print(f"\nLoading face detection model: {args.face_weights}")
    model = YOLO(args.face_weights)
    
    # Create output directories
    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, 'crops'))
    
    all_crops_data = []
    
    # Process each split
    for split in args.splits:
        crops_data = process_dataset_split(
            args.data_dir, split, model, args.output_dir, target_size, args.conf_threshold
        )
        all_crops_data.extend(crops_data)
        
        # Save crops CSV for this split
        if crops_data:
            csv_path = os.path.join(args.output_dir, f'{split}_crops.csv')
            df = pd.DataFrame(crops_data)
            df.to_csv(csv_path, index=False)
            print(f"  ✓ Saved {len(crops_data)} crops to {csv_path}")
        else:
            print(f"  ⚠️  No crops processed for {split} set")
    
    # Save combined crops CSV
    if all_crops_data:
        all_csv_path = os.path.join(args.output_dir, 'all_crops.csv')
        df_all = pd.DataFrame(all_crops_data)
        df_all.to_csv(all_csv_path, index=False)
        print(f"\n✓ Total crops saved: {len(all_crops_data)}")
        print(f"✓ Combined CSV saved: {all_csv_path}")
        
        # Print statistics
        print("\n" + "="*60)
        print("CROPPING STATISTICS (2 Classes)")
        print("="*60)
        for split in args.splits:
            split_crops = [c for c in all_crops_data if split in c['crop_path']]
            if split_crops:
                classes = set(c['class_name'] for c in split_crops)
                print(f"{split.upper()}:")
                for class_name in sorted(classes):
                    count = len([c for c in split_crops if c['class_name'] == class_name])
                    print(f"  {class_name}: {count} faces")
                print(f"  Total: {len(split_crops)} faces")
    else:
        print("\n⚠️  No crops processed!")
    
    print("\n" + "="*60)
    print("Face Detection and Cropping Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
