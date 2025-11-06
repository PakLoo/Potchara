#!/usr/bin/env python3
"""
Test MobileNetV2 model on test dataset (2 classes)
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
import argparse
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

# Import models with fallback
try:
    from torchvision import models
except ImportError:
    print("Warning: torchvision.models not available, using alternative approach")
    models = None

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not available, using simple progress")
    def tqdm(iterable, desc="", **kwargs):
        return iterable

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


def test_model(model, test_loader, device, class_names):
    """Test the model on test dataset"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Get predictions
            _, predicted = torch.max(output, 1)
            probs = torch.softmax(output, dim=1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return all_preds, all_labels, all_probs


def print_class_metrics(y_true, y_pred, class_names):
    """Print per-class metrics"""
    print("\nPer-class accuracy:")
    for i, class_name in enumerate(class_names):
        class_mask = np.array(y_true) == i
        if np.sum(class_mask) > 0:
            class_acc = np.sum(np.array(y_pred)[class_mask] == i) / np.sum(class_mask)
            print(f"  {class_name}: {class_acc:.4f}")


def print_confusion_matrix(y_true, y_pred, class_names):
    """Print confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print("Predicted ->")
    print("Actual ->", end="")
    for class_name in class_names:
        print(f"{class_name:>12}", end="")
    print()
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:>12}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i][j]:>12}", end="")
        print()


def save_test_results(results_dir, y_true, y_pred, y_probs, class_names, accuracy):
    """Save test results to files"""
    ensure_dir(results_dir)
    
    # Save predictions
    results_df = pd.DataFrame({
        'true_label': [class_names[i] for i in y_true],
        'predicted_label': [class_names[i] for i in y_pred],
        'correct': [y_true[i] == y_pred[i] for i in range(len(y_true))]
    })
    
    # Add probabilities
    for i, class_name in enumerate(class_names):
        results_df[f'prob_{class_name}'] = [probs[i] for probs in y_probs]
    
    results_path = os.path.join(results_dir, 'test_results_2class.csv')
    results_df.to_csv(results_path, index=False)
    print(f"✓ Test results saved: {results_path}")
    
    # Save summary
    summary_path = os.path.join(results_dir, 'test_summary_2class.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("MobileNetV2 Test Results (2 Classes)\n")
        f.write("="*60 + "\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        
        f.write("Per-class accuracy:\n")
        for i, class_name in enumerate(class_names):
            class_mask = np.array(y_true) == i
            if np.sum(class_mask) > 0:
                class_acc = np.sum(np.array(y_pred)[class_mask] == i) / np.sum(class_mask)
                f.write(f"  {class_name}: {class_acc:.4f}\n")
        
        f.write("\nConfusion Matrix:\n")
        cm = confusion_matrix(y_true, y_pred)
        f.write("Predicted ->\n")
        f.write("Actual ->")
        for class_name in class_names:
            f.write(f"{class_name:>12}")
        f.write("\n")
        
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:>12}")
            for j in range(len(class_names)):
                f.write(f"{cm[i][j]:>12}")
            f.write("\n")
        
        f.write("="*60 + "\n")
    
    print(f"✓ Test summary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Test MobileNetV2 model (2 classes)')
    parser.add_argument('--data_dir', type=str, default='data3/processed/crops',
                       help='Path to cropped faces directory')
    parser.add_argument('--model_path', type=str, default='models3/image_baseline/mobilenetv2_best_2class.pt',
                       help='Path to trained model')
    parser.add_argument('--results_dir', type=str, default='Results4/testing',
                       help='Output directory for test results')
    parser.add_argument('--img_size', type=int, default=500,
                       help='Input image size (should match training data)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for testing')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MobileNetV2 Model Testing (2 Classes)")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Model path: {args.model_path}")
    print(f"Results directory: {args.results_dir}")
    print(f"Image size: {args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Classes: no_mask, mask")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU instead")
    
    print(f"Using device: {device}")
    
    # Check if test set exists
    test_path = os.path.join(args.data_dir, 'test')
    if not os.path.exists(test_path):
        print(f"Error: Test directory not found: {test_path}")
        return
    
    # Data transforms
    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = ImageFolder(test_path, transform=test_transform)
    print(f"Test: {len(test_dataset)} images")
    print(f"Classes: {test_dataset.classes}")
    
    # Data loader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    print("\nLoading model...")
    num_classes = len(test_dataset.classes)
    model = create_model(num_classes=num_classes, pretrained=False)
    
    # Load trained weights
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()
    print(f"Model loaded: {args.model_path}")
    
    # Test model
    print("\n" + "="*60)
    print("Testing Model")
    print("="*60)
    
    y_pred, y_true, y_probs = test_model(model, test_loader, device, test_dataset.classes)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n🎯 Overall Accuracy: {accuracy:.4f}")
    
    # Print detailed metrics
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT (2 Classes)")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=test_dataset.classes))
    
    # Print confusion matrix
    print_confusion_matrix(y_true, y_pred, test_dataset.classes)
    
    # Print per-class accuracy
    print_class_metrics(y_true, y_pred, test_dataset.classes)
    
    # Save results
    save_test_results(args.results_dir, y_true, y_pred, y_probs, test_dataset.classes, accuracy)
    
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
