#!/usr/bin/env python3
"""
Extract facial landmarks and features from cropped faces using MediaPipe FaceMesh
For 2-class classification (no_mask vs mask)
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm
import argparse
import warnings

from utils.paths import ensure_dir

# Suppress MediaPipe warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


def extract_landmarks(image_path):
    """Extract facial landmarks from a single image"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None, None
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,  # Only process 1 face
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(image_rgb)
            
            if results.multi_face_landmarks:
                # Get the first face
                face_landmarks = results.multi_face_landmarks[0]
                
                # Extract landmark coordinates with visibility
                landmarks = []
                for landmark in face_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                
                # Extract key facial features
                features = extract_facial_features(face_landmarks, image.shape)
                
                return np.array(landmarks), features
            else:
                return None, None
                
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None


def extract_facial_features(face_landmarks, image_shape):
    """Extract specific facial features from landmarks"""
    h, w = image_shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    def get_landmark_coords(landmark_idx):
        landmark = face_landmarks.landmark[landmark_idx]
        return int(landmark.x * w), int(landmark.y * h)
    
    features = {}
    
    # Eye features
    # Left eye
    left_eye_left = get_landmark_coords(33)
    left_eye_right = get_landmark_coords(133)
    left_eye_top = get_landmark_coords(159)
    left_eye_bottom = get_landmark_coords(145)
    
    # Right eye
    right_eye_left = get_landmark_coords(362)
    right_eye_right = get_landmark_coords(263)
    right_eye_top = get_landmark_coords(386)
    right_eye_bottom = get_landmark_coords(374)
    
    # Calculate eye dimensions
    left_eye_width = abs(left_eye_right[0] - left_eye_left[0])
    left_eye_height = abs(left_eye_bottom[1] - left_eye_top[1])
    right_eye_width = abs(right_eye_right[0] - right_eye_left[0])
    right_eye_height = abs(right_eye_bottom[1] - right_eye_top[1])
    
    features['left_eye_width'] = left_eye_width
    features['left_eye_height'] = left_eye_height
    features['right_eye_width'] = right_eye_width
    features['right_eye_height'] = right_eye_height
    features['left_eye_aspect_ratio'] = left_eye_height / max(left_eye_width, 1)
    features['right_eye_aspect_ratio'] = right_eye_height / max(right_eye_width, 1)
    
    # Nose features
    nose_tip = get_landmark_coords(1)
    nose_bridge = get_landmark_coords(6)
    nose_left = get_landmark_coords(31)
    nose_right = get_landmark_coords(35)
    
    nose_width = abs(nose_right[0] - nose_left[0])
    nose_height = abs(nose_tip[1] - nose_bridge[1])
    
    features['nose_width'] = nose_width
    features['nose_height'] = nose_height
    features['nose_aspect_ratio'] = nose_height / max(nose_width, 1)
    
    # Mouth features
    mouth_left = get_landmark_coords(61)
    mouth_right = get_landmark_coords(291)
    mouth_top = get_landmark_coords(13)
    mouth_bottom = get_landmark_coords(14)
    
    mouth_width = abs(mouth_right[0] - mouth_left[0])
    mouth_height = abs(mouth_bottom[1] - mouth_top[1])
    
    features['mouth_width'] = mouth_width
    features['mouth_height'] = mouth_height
    features['mouth_aspect_ratio'] = mouth_height / max(mouth_width, 1)
    
    # Face shape features
    face_left = get_landmark_coords(172)
    face_right = get_landmark_coords(397)
    face_top = get_landmark_coords(10)
    face_bottom = get_landmark_coords(152)
    
    face_width = abs(face_right[0] - face_left[0])
    face_height = abs(face_bottom[1] - face_top[1])
    
    features['face_width'] = face_width
    features['face_height'] = face_height
    features['face_aspect_ratio'] = face_height / max(face_width, 1)
    
    # Eye-mouth distance
    left_eye_center = ((left_eye_left[0] + left_eye_right[0]) // 2, 
                      (left_eye_top[1] + left_eye_bottom[1]) // 2)
    right_eye_center = ((right_eye_left[0] + right_eye_right[0]) // 2, 
                       (right_eye_top[1] + right_eye_bottom[1]) // 2)
    mouth_center = ((mouth_left[0] + mouth_right[0]) // 2, 
                   (mouth_top[1] + mouth_bottom[1]) // 2)
    
    eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                  (left_eye_center[1] + right_eye_center[1]) // 2)
    
    eye_mouth_distance = abs(eye_center[1] - mouth_center[1])
    features['eye_mouth_distance'] = eye_mouth_distance
    
    # Visibility features (key for mask detection)
    features['mouth_visibility'] = face_landmarks.landmark[13].visibility  # Top lip
    features['nose_visibility'] = face_landmarks.landmark[1].visibility   # Nose tip
    features['chin_visibility'] = face_landmarks.landmark[152].visibility # Chin
    
    # Average visibility of mouth area landmarks (key mouth landmarks)
    mouth_landmarks = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467]
    mouth_visibility_avg = np.mean([face_landmarks.landmark[i].visibility for i in mouth_landmarks])
    features['mouth_visibility_avg'] = mouth_visibility_avg
    
    # Normalize features by face size
    face_area = face_width * face_height
    if face_area > 0:
        for key in features:
            if 'ratio' not in key and 'distance' not in key and 'visibility' not in key:
                features[key] = features[key] / np.sqrt(face_area)
    
    return features


def process_crops_csv(crops_csv_path, output_dir):
    """Process all crops from a CSV file"""
    print(f"Processing crops from: {crops_csv_path}")
    
    # Load crops data
    if not os.path.exists(crops_csv_path):
        print(f"Warning: Crops CSV not found: {crops_csv_path}")
        return []
    
    df = pd.read_csv(crops_csv_path)
    print(f"Found {len(df)} crops to process")
    
    landmarks_data = []
    features_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting landmarks"):
        crop_path = row['crop_path']
        
        if not os.path.exists(crop_path):
            print(f"Warning: Crop image not found: {crop_path}")
            continue
        
        # Extract landmarks and features
        landmarks, features = extract_landmarks(crop_path)
        
        if landmarks is not None and features is not None:
            # Add metadata
            landmarks_data.append({
                'crop_path': crop_path,
                'class_name': row['class_name'],
                'face_id': row['face_id'],
                'landmarks': landmarks.tolist()
            })
            
            # Add features with metadata
            feature_row = {
                'crop_path': crop_path,
                'class_name': row['class_name'],
                'face_id': row['face_id']
            }
            feature_row.update(features)
            features_data.append(feature_row)
        else:
            print(f"Warning: Could not extract landmarks from {crop_path}")
    
    # Save landmarks data
    if landmarks_data:
        landmarks_df = pd.DataFrame(landmarks_data)
        landmarks_path = os.path.join(output_dir, 'landmarks.csv')
        landmarks_df.to_csv(landmarks_path, index=False)
        print(f"✓ Saved landmarks to: {landmarks_path}")
    
    # Save features data
    if features_data:
        features_df = pd.DataFrame(features_data)
        features_path = os.path.join(output_dir, 'features.csv')
        features_df.to_csv(features_path, index=False)
        print(f"✓ Saved features to: {features_path}")
    
    return features_data


def main():
    parser = argparse.ArgumentParser(description='Extract landmarks and features from cropped faces (2 classes)')
    parser.add_argument('--crops_dir', type=str, default='data3/processed',
                       help='Directory containing crops CSV files')
    parser.add_argument('--output_dir', type=str, default='data3/processed/landmarks',
                       help='Output directory for landmarks and features')
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'],
                       help='Dataset splits to process')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Facial Landmarks and Features Extraction (2 Classes)")
    print("="*60)
    print(f"Crops directory: {args.crops_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Splits to process: {args.splits}")
    print(f"Classes: no_mask, mask")
    
    # Create output directory
    ensure_dir(args.output_dir)
    
    all_features_data = []
    
    # Process each split
    for split in args.splits:
        print(f"\nProcessing {split} split...")
        crops_csv_path = os.path.join(args.crops_dir, f'{split}_crops.csv')
        
        features_data = process_crops_csv(crops_csv_path, args.output_dir)
        all_features_data.extend(features_data)
    
    # Save combined features
    if all_features_data:
        all_features_df = pd.DataFrame(all_features_data)
        all_features_path = os.path.join(args.output_dir, 'all_features.csv')
        all_features_df.to_csv(all_features_path, index=False)
        print(f"\n✓ Total features extracted: {len(all_features_data)}")
        print(f"✓ Combined features saved: {all_features_path}")
        
        # Print statistics
        print("\n" + "="*60)
        print("FEATURE EXTRACTION STATISTICS (2 Classes)")
        print("="*60)
        for split in args.splits:
            split_features = [f for f in all_features_data if split in f['crop_path']]
            if split_features:
                classes = set(f['class_name'] for f in split_features)
                print(f"{split.upper()}:")
                for class_name in sorted(classes):
                    count = len([f for f in split_features if f['class_name'] == class_name])
                    print(f"  {class_name}: {count} faces")
                print(f"  Total: {len(split_features)} faces")
    else:
        print("\n⚠️  No features extracted!")
    
    print("\n" + "="*60)
    print("Landmark and Feature Extraction Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
