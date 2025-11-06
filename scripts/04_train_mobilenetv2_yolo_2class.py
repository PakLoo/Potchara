#!/usr/bin/env python3
"""
Train MobileNetV2 classifier using cropped faces from YOLO dataset (2 classes)
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import argparse

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


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device, split_name="Validation"):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in tqdm(loader, desc=split_name):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def print_class_metrics(y_true, y_pred, class_names):
    """Print per-class metrics"""
    print("\nPer-class accuracy:")
    for i, class_name in enumerate(class_names):
        class_mask = np.array(y_true) == i
        if np.sum(class_mask) > 0:
            class_acc = np.sum(np.array(y_pred)[class_mask] == i) / np.sum(class_mask)
            print(f"  {class_name}: {class_acc:.4f}")


def save_training_log(log_path, train_losses, train_accs, val_losses, val_accs):
    """Save training log to CSV"""
    import pandas as pd
    
    epochs = range(1, len(train_losses) + 1)
    log_data = {
        'epoch': epochs,
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs
    }
    
    df = pd.DataFrame(log_data)
    df.to_csv(log_path, index=False)


def main():
    parser = argparse.ArgumentParser(description='Train MobileNetV2 classifier (2 classes)')
    parser.add_argument('--data_dir', type=str, default='data3/processed/crops',
                       help='Path to cropped faces directory')
    parser.add_argument('--output_dir', type=str, default='models3/image_baseline',
                       help='Output directory for trained models')
    parser.add_argument('--results_dir', type=str, default='Results3/training',
                       help='Output directory for training results')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--img_size', type=int, default=500,
                       help='Input image size (should match training data)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MobileNetV2 Training with YOLO Dataset (2 Classes)")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Results directory: {args.results_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Image size: {args.img_size}")
    print(f"Device: {args.device}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Classes: no_mask, mask")
    
    # Create output directories
    ensure_dir(args.output_dir)
    ensure_dir(args.results_dir)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU instead")
    
    print(f"Using device: {device}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = ImageFolder(os.path.join(args.data_dir, 'train'), transform=train_transform)
    val_dataset = ImageFolder(os.path.join(args.data_dir, 'valid'), transform=val_transform)
    
    # Check if test set exists
    test_dataset = None
    test_path = os.path.join(args.data_dir, 'test')
    if os.path.exists(test_path):
        test_dataset = ImageFolder(test_path, transform=val_transform)
        print(f"Test:  {len(test_dataset)} images")
    else:
        print("Test:  Not found (will skip test evaluation)")
    
    print(f"Train: {len(train_dataset)} images")
    print(f"Valid: {len(val_dataset)} images")
    print(f"Classes: {train_dataset.classes}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Test loader (if test set exists)
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    print("\nCreating model...")
    num_classes = len(train_dataset.classes)
    model = create_model(num_classes=num_classes, pretrained=args.pretrained)
    model = model.to(device)
    
    print(f"Model: MobileNetV2 with {num_classes} classes")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    best_val_acc = 0.0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device, "Validation")
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.output_dir, 'mobilenetv2_best_2class.pt')
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ New best model saved: {best_model_path}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'mobilenetv2_final_2class.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✓ Final model saved: {final_model_path}")
    
    # Save training log
    log_path = os.path.join(args.results_dir, "training_log_2class.csv")
    save_training_log(log_path, train_losses, train_accs, val_losses, val_accs)
    print(f"✓ Training log saved: {log_path}")
    
    # Test evaluation (if test set exists)
    if test_loader is not None:
        print("\n" + "="*60)
        print("Evaluating on Test Set")
        print("="*60)
        
        # Load best model
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=False))
        test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device, "Test")
        
        print(f"\n🎯 Test Accuracy: {test_acc:.2f}%")
        
        # Print detailed metrics
        print_class_metrics(test_labels, test_preds, train_dataset.classes)
    else:
        print("\n" + "="*60)
        print("No Test Set Available - Skipping Test Evaluation")
        print("="*60)
        print("Using validation accuracy as final metric")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Save summary
    summary_path = os.path.join(args.results_dir, "summary_2class.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Training Summary (2 Classes)\n")
        f.write("="*60 + "\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        if test_loader is not None:
            f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        else:
            f.write("Test Accuracy: N/A (No test set)\n")
        f.write(f"Model: MobileNetV2\n")
        f.write(f"Classes: {num_classes} (no_mask, mask)\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Image size: {args.img_size}\n")
        f.write(f"Best model path: {best_model_path}\n")
        f.write("="*60 + "\n")
    
    print(f"\n✓ Summary saved: {summary_path}")
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
