#!/usr/bin/env python3
"""
Train ML classifiers using facial landmarks and features for 2-class classification
Supports Random Forest and XGBoost
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import argparse
from tqdm import tqdm

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

# Try to import SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: SMOTE not available. Install with: pip install imbalanced-learn")

from utils.paths import ensure_dir


def load_features_data(features_csv_path):
    """Load features data from CSV"""
    if not os.path.exists(features_csv_path):
        print(f"Error: Features file not found: {features_csv_path}")
        return None, None
    
    df = pd.read_csv(features_csv_path)
    print(f"Loaded {len(df)} samples from {features_csv_path}")
    
    # Separate features and labels
    feature_columns = [col for col in df.columns if col not in ['crop_path', 'class_name', 'face_id']]
    X = df[feature_columns].values
    y = df['class_name'].values
    
    # Convert class names to numeric labels (2 classes)
    class_names = sorted(df['class_name'].unique())
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    y_numeric = np.array([class_to_id[label] for label in y])
    
    print(f"Features: {len(feature_columns)}")
    print(f"Classes: {class_names}")
    print(f"Class distribution:")
    for class_name in class_names:
        count = np.sum(y == class_name)
        print(f"  {class_name}: {count} samples")
    
    return X, y_numeric, class_names, feature_columns


def train_classifier(X, y, classifier_type='rf', test_size=0.2, random_state=42, use_smote=False):
    """Train a classifier with the given data"""
    print(f"\nTraining {classifier_type.upper()} classifier...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE if requested
    if use_smote and SMOTE_AVAILABLE:
        print("Applying SMOTE to balance classes...")
        smote = SMOTE(random_state=random_state)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        print(f"After SMOTE - Class distribution: {np.bincount(y_train)}")
    elif use_smote and not SMOTE_AVAILABLE:
        print("Warning: SMOTE not available, training without balancing")
    
    # Initialize classifier
    if classifier_type == 'rf':
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
    elif classifier_type == 'xgb' and XGBOOST_AVAILABLE:
        classifier = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=-1,
            scale_pos_weight=1,  # Balanced for 2 classes
            class_weight='balanced'  # Use balanced class weight
        )
    elif classifier_type == 'xgb' and not XGBOOST_AVAILABLE:
        print("Error: XGBoost not available. Falling back to Random Forest.")
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    # Train classifier
    print("Training classifier...")
    classifier.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    y_pred = classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Cross-validation
    print("Performing cross-validation...")
    cv_scores = cross_val_score(classifier, X_train_scaled, y_train, cv=5)
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return classifier, scaler, X_test_scaled, y_test, y_pred, accuracy


def hyperparameter_tuning(X, y, classifier_type='rf', n_jobs=-1, use_smote=False):
    """Perform hyperparameter tuning"""
    print(f"\nPerforming hyperparameter tuning for {classifier_type.upper()}...")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply SMOTE if requested
    if use_smote and SMOTE_AVAILABLE:
        print("Applying SMOTE to balance classes...")
        smote = SMOTE(random_state=42)
        X_scaled, y = smote.fit_resample(X_scaled, y)
        print(f"After SMOTE - Class distribution: {np.bincount(y)}")
    elif use_smote and not SMOTE_AVAILABLE:
        print("Warning: SMOTE not available, tuning without balancing")
    
    # Define parameter grids
    if classifier_type == 'rf':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        base_classifier = RandomForestClassifier(random_state=42, n_jobs=n_jobs)
    elif classifier_type == 'xgb' and XGBOOST_AVAILABLE:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        base_classifier = xgb.XGBClassifier(random_state=42, n_jobs=n_jobs, class_weight='balanced')
    elif classifier_type == 'xgb' and not XGBOOST_AVAILABLE:
        print("Error: XGBoost not available. Using Random Forest instead.")
        return hyperparameter_tuning(X, y, 'rf', n_jobs)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    # Grid search
    grid_search = GridSearchCV(
        base_classifier,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=1
    )
    
    grid_search.fit(X_scaled, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, scaler


def evaluate_classifier(classifier, X_test, y_test, class_names):
    """Evaluate classifier performance"""
    y_pred = classifier.predict(X_test)
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT (2 Classes)")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    print("\nCONFUSION MATRIX:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Calculate per-class accuracy
    print("\nPer-class accuracy:")
    for i, class_name in enumerate(class_names):
        class_mask = y_test == i
        if np.sum(class_mask) > 0:
            class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
            print(f"  {class_name}: {class_acc:.4f}")
    
    return y_pred


def main():
    parser = argparse.ArgumentParser(description='Train ML classifier using facial landmarks (2 classes)')
    parser.add_argument('--features_dir', type=str, default='data3/processed/landmarks',
                       help='Directory containing features CSV files')
    parser.add_argument('--output_dir', type=str, default='models3/landmark_ml',
                       help='Output directory for trained models')
    parser.add_argument('--classifier', type=str, default='rf', choices=['rf', 'xgb'],
                       help='Classifier type: rf (Random Forest), xgb (XGBoost)')
    parser.add_argument('--tune_hyperparams', action='store_true',
                       help='Perform hyperparameter tuning')
    parser.add_argument('--use_smote', action='store_true',
                       help='Use SMOTE to balance classes')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size ratio')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Landmark ML Classifier Training (2 Classes)")
    print("="*60)
    print(f"Features directory: {args.features_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Classifier: {args.classifier.upper()}")
    print(f"Hyperparameter tuning: {args.tune_hyperparams}")
    print(f"Use SMOTE: {args.use_smote}")
    print(f"Classes: no_mask, mask")
    
    # Create output directory
    ensure_dir(args.output_dir)
    
    # Load features data
    features_path = os.path.join(args.features_dir, 'all_features.csv')
    X, y, class_names, feature_columns = load_features_data(features_path)
    
    if X is None:
        print("Error: Could not load features data")
        return
    
    # Train classifier
    if args.tune_hyperparams:
        classifier, scaler = hyperparameter_tuning(X, y, args.classifier, use_smote=args.use_smote)
        
        # Final evaluation with best parameters
        X_scaled = scaler.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )
        classifier.fit(X_train, y_train)
        y_pred = evaluate_classifier(classifier, X_test, y_test, class_names)
    else:
        classifier, scaler, X_test, y_test, y_pred, accuracy = train_classifier(
            X, y, args.classifier, args.test_size, args.random_state, use_smote=args.use_smote
        )
        evaluate_classifier(classifier, X_test, y_test, class_names)
    
    # Save model and scaler
    model_path = os.path.join(args.output_dir, f'best_landmark_{args.classifier}_2class.pkl')
    scaler_path = os.path.join(args.output_dir, f'scaler_{args.classifier}_2class.pkl')
    
    joblib.dump(classifier, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\n✓ Model saved: {model_path}")
    print(f"✓ Scaler saved: {scaler_path}")
    
    # Save metadata
    metadata = {
        'classifier_type': args.classifier,
        'class_names': class_names,
        'feature_columns': feature_columns,
        'n_features': len(feature_columns),
        'n_classes': len(class_names),
        'test_accuracy': accuracy if not args.tune_hyperparams else None
    }
    
    metadata_path = os.path.join(args.output_dir, f'metadata_{args.classifier}_2class.pkl')
    joblib.dump(metadata, metadata_path)
    print(f"✓ Metadata saved: {metadata_path}")
    
    print("\n" + "="*60)
    print("Landmark ML Classifier Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
