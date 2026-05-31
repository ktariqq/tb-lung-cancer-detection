"""
Transfer learning models for multi-class medical image classification.
Implements ResNet50, VGG16, and EfficientNet architectures.
"""

import torch
import torch.nn as nn
from torchvision import models
import timm


class ResNet50Classifier(nn.Module):
    """
    ResNet50-based classifier for multi-class medical imaging.
    Uses pretrained ImageNet weights with custom classification head.
    """
    def __init__(self, num_classes=3, pretrained=True, freeze_base=True):
        """
        Args:
            num_classes: Number of output classes (e.g., 3 for TB/Cancer/Normal)
            pretrained: Use ImageNet pretrained weights
            freeze_base: Freeze convolutional layers initially
        """
        super(ResNet50Classifier, self).__init__()
        
        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Freeze base layers if specified
        if freeze_base:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Get number of features from last layer
        num_features = self.resnet.fc.in_features
        
        # Replace classification head with custom layers
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        
        # Always train the new classification head
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.resnet(x)
    
    def unfreeze_layers(self, num_layers=10):
        """
        Unfreeze top layers for fine-tuning.
        
        Args:
            num_layers: Number of layers from end to unfreeze
        """
        # Get all layers
        layers = list(self.resnet.children())[:-1]  # Exclude FC layer
        
        # Unfreeze last num_layers
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        print(f"Unfroze top {num_layers} layers for fine-tuning")


class VGG16Classifier(nn.Module):
    """
    VGG16-based classifier for multi-class medical imaging.
    """
    def __init__(self, num_classes=3, pretrained=True, freeze_base=True):
        super(VGG16Classifier, self).__init__()
        
        # Load pretrained VGG16
        self.vgg = models.vgg16(pretrained=pretrained)
        
        # Freeze convolutional layers
        if freeze_base:
            for param in self.vgg.features.parameters():
                param.requires_grad = False
        
        # Get number of features
        num_features = self.vgg.classifier[0].in_features
        
        # Custom classification head
        self.vgg.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.vgg(x)


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B3 classifier for multi-class medical imaging.
    State-of-the-art architecture with excellent accuracy/efficiency trade-off.
    """
    def __init__(self, num_classes=3, pretrained=True, freeze_base=True):
        super(EfficientNetClassifier, self).__init__()
        
        # Load pretrained EfficientNet-B3 using timm library
        self.efficientnet = timm.create_model(
            'efficientnet_b3',
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        
        # Freeze base if specified
        if freeze_base:
            for param in self.efficientnet.parameters():
                param.requires_grad = False
        
        # Get number of features
        num_features = self.efficientnet.num_features
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.efficientnet(x)
        return self.classifier(features)


def create_model(architecture='resnet50', num_classes=3, pretrained=True, freeze_base=True):
    """
    Factory function to create transfer learning models.
    
    Args:
        architecture: 'resnet50', 'vgg16', or 'efficientnet'
        num_classes: Number of output classes
        pretrained: Use ImageNet weights
        freeze_base: Freeze convolutional layers
    
    Returns:
        Model instance
    """
    if architecture.lower() == 'resnet50':
        model = ResNet50Classifier(num_classes, pretrained, freeze_base)
    elif architecture.lower() == 'vgg16':
        model = VGG16Classifier(num_classes, pretrained, freeze_base)
    elif architecture.lower() == 'efficientnet':
        model = EfficientNetClassifier(num_classes, pretrained, freeze_base)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model


def get_model_info(model):
    """
    Get detailed information about model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'trainable_percentage': (trainable_params / total_params) * 100
    }


def print_model_summary(model, model_name):
    """
    Print formatted model summary.
    
    Args:
        model: PyTorch model
        model_name: Name for display
    """
    info = get_model_info(model)
    
    print("\n" + "="*70)
    print(f"{model_name} MODEL SUMMARY")
    print("="*70)
    print(f"Total Parameters:      {info['total_params']:,}")
    print(f"Trainable Parameters:  {info['trainable_params']:,}")
    print(f"Frozen Parameters:     {info['frozen_params']:,}")
    print(f"Trainable Percentage:  {info['trainable_percentage']:.2f}%")
    print("="*70 + "\n")
