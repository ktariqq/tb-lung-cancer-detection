"""
Data preprocessing utilities including SMOTE for handling class imbalance.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import cv2


class MedicalImageDataset(Dataset):
    """
    Custom PyTorch Dataset for medical images with multiple classes.
    """
    def __init__(self, image_paths, labels, transform=None, img_size=(224, 224)):
        """
        Args:
            image_paths: List of paths to images
            labels: List of integer labels
            transform: Optional torchvision transforms
            img_size: Target image size
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.img_size = img_size
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Resize if no transform specified
        if self.transform is None:
            image = image.resize(self.img_size)
            image = transforms.ToTensor()(image)
        else:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label


def load_dataset_from_folders(data_dirs, class_names):
    """
    Load medical imaging dataset from folder structure.
    
    Expected structure:
    data_dir/
    ├── class_1/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── class_2/
        ├── img3.jpg
        └── img4.jpg
    
    Args:
        data_dirs: List of directories containing class folders
        class_names: List of class names (order determines label encoding)
    
    Returns:
        image_paths, labels, class_to_idx mapping
    """
    image_paths = []
    labels = []
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    
    for data_dir in data_dirs:
        for class_name in class_names:
            class_dir = os.path.join(data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Directory not found: {class_dir}")
                continue
            
            # Get all image files
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                    img_path = os.path.join(class_dir, img_file)
                    image_paths.append(img_path)
                    labels.append(class_to_idx[class_name])
    
    print(f"Total images loaded: {len(image_paths)}")
    print(f"Classes: {class_names}")
    print(f"Class distribution:")
    for class_name, idx in class_to_idx.items():
        count = labels.count(idx)
        print(f"  {class_name}: {count} images ({count/len(labels)*100:.1f}%)")
    print("="*70 + "\n")
    
    return image_paths, labels, class_to_idx


def apply_smote_to_features(X, y, random_state=42):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance classes.
    
    Note: For images, we extract features first, apply SMOTE, then use indices.
    This is a simplified version - real implementation would use image features.
    
    Args:
        X: Feature array (num_samples, num_features)
        y: Labels (num_samples,)
        random_state: Random seed
    
    Returns:
        X_resampled, y_resampled
    """
    print("\nApplying SMOTE for class balancing...")
    print(f"Original distribution: {np.bincount(y)}")
    
    # Apply SMOTE
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print(f"Resampled distribution: {np.bincount(y_resampled)}")
    print(f"Original samples: {len(y)}")
    print(f"Resampled samples: {len(y_resampled)}\n")
    
    return X_resampled, y_resampled


def create_data_loaders(image_paths, labels, batch_size=32, val_split=0.15, 
                       test_split=0.15, random_state=42, num_workers=2):
    """
    Create train, validation, and test data loaders with augmentation.
    
    Args:
        image_paths: List of image paths
        labels: List of labels
        batch_size: Batch size
        val_split: Validation set proportion
        test_split: Test set proportion
        random_state: Random seed
        num_workers: Number of parallel workers
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Split data: train + (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, 
        test_size=(val_split + test_split),
        stratify=labels,
        random_state=random_state
    )
    
    # Split temp into val and test
    val_ratio = val_split / (val_split + test_split)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio),
        stratify=y_temp,
        random_state=random_state
    )
    
    print("Dataset split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples\n")
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    # No augmentation for val/test
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = MedicalImageDataset(X_train, y_train, transform=train_transform)
    val_dataset = MedicalImageDataset(X_val, y_val, transform=val_test_transform)
    test_dataset = MedicalImageDataset(X_test, y_test, transform=val_test_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def calculate_class_weights(labels):
    """
    Calculate class weights for imbalanced dataset (inverse frequency).
    
    Args:
        labels: List of labels
    
    Returns:
        Tensor of class weights
    """
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    # Inverse frequency weighting
    class_weights = total_samples / (num_classes * class_counts)
    
    print("\nClass weights (for loss function):")
    for i, weight in enumerate(class_weights):
        print(f"  Class {i}: {weight:.4f} (count: {class_counts[i]})")
    
    return torch.FloatTensor(class_weights)
