"""
Inference script for making predictions on new images.
"""

import torch
import sys
sys.path.append('..')
from models.transfer_learning_models import create_model
from torchvision import transforms
from PIL import Image
import argparse


def load_model(model_path, architecture='resnet50', num_classes=3, device='cuda'):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to saved model (.pth file)
        architecture: Model architecture
        num_classes: Number of classes
        device: 'cuda' or 'cpu'
    
    Returns:
        Loaded model
    """
    model = create_model(architecture, num_classes, pretrained=False, freeze_base=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, img_size=(224, 224)):
    """
    Preprocess single image for inference.
    
    Args:
        image_path: Path to image
        img_size: Target size
    
    Returns:
        Preprocessed tensor
    """
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def predict(model, image_tensor, class_names, device='cuda'):
    """
    Make prediction on single image.
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        class_names: List of class names
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary with prediction results
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
    
    predicted_class = predicted_class.item()
    confidence = confidence.item()
    
    return {
        'predicted_class': class_names[predicted_class],
        'predicted_index': predicted_class,
        'confidence': confidence * 100,
        'all_probabilities': {
            class_names[i]: probs[0, i].item() * 100 
            for i in range(len(class_names))
        }
    }


def main():
    """Command-line interface for inference."""
    parser = argparse.ArgumentParser(description='TB & Lung Cancer Detection Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--architecture', type=str, default='resnet50',
                       choices=['resnet50', 'vgg16', 'efficientnet'],
                       help='Model architecture')
    parser.add_argument('--classes', type=str, nargs='+',
                       default=['Normal', 'TB', 'Lung Cancer'],
                       help='Class names')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"\nLoading model from {args.model}...")
    model = load_model(args.model, args.architecture, len(args.classes), args.device)
    
    print(f"Processing image: {args.image}...")
    image_tensor = preprocess_image(args.image)
    
    print("Making prediction...\n")
    result = predict(model, image_tensor, args.classes, args.device)
    
    print("="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print(f"\nProbabilities for all classes:")
    for class_name, prob in result['all_probabilities'].items():
        print(f"  {class_name}: {prob:.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()
