"""
Visualization utilities for multi-class classification results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import torch


def plot_training_history(history, save_path='../results/training_history.png'):
    """
    Plot training and validation metrics.
    
    Args:
        history: Dictionary with 'train_loss', 'train_acc', 'val_loss', 'val_acc'
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training history saved to {save_path}")


def plot_confusion_matrix(cm, class_names, normalize=False, 
                         save_path='../results/confusion_matrix.png'):
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        normalize: Whether to normalize by row (true label)
        save_path: Path to save plot
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved to {save_path}")


def plot_per_class_metrics(metrics_dict, class_names, 
                          save_path='../results/per_class_metrics.png'):
    """
    Plot per-class precision, recall, and F1-score as bar charts.
    
    Args:
        metrics_dict: Dictionary with 'precision_per_class', 'recall_per_class', 'f1_per_class'
        class_names: List of class names
        save_path: Path to save plot
    """
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, metrics_dict['precision_per_class'], width, 
                   label='Precision', color='steelblue')
    bars2 = ax.bar(x, metrics_dict['recall_per_class'], width,
                   label='Recall', color='orange')
    bars3 = ax.bar(x + width, metrics_dict['f1_per_class'], width,
                   label='F1-Score', color='green')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Per-class metrics saved to {save_path}")


def plot_multiclass_roc_curves(labels, probabilities, class_names, num_classes,
                               save_path='../results/roc_curves.png'):
    """
    Plot ROC curves for multi-class classification (One-vs-Rest).
    
    Args:
        labels: True labels
        probabilities: Predicted probabilities (n_samples, n_classes)
        class_names: List of class names
        num_classes: Number of classes
        save_path: Path to save plot
    """
    # Binarize labels for OvR
    y_binary = label_binarize(labels, classes=range(num_classes))
    
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, num_classes))
    
    # Plot ROC curve for each class
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_binary[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color, lw=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-Class ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"ROC curves saved to {save_path}")


def plot_model_comparison(comparison_df, save_path='../results/model_comparison.png'):
    """
    Plot bar chart comparing multiple models.
    
    Args:
        comparison_df: DataFrame with model comparison metrics
        save_path: Path to save plot
    """
    # Extract numeric values from percentage strings
    metrics = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)']
    
    data = []
    for metric in metrics:
        if metric in comparison_df.columns:
            values = comparison_df[metric].str.rstrip('%').astype(float).values
            data.append(values)
    
    models = comparison_df['Model'].values
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['steelblue', 'orange', 'green', 'red']
    for i, (metric, values, color) in enumerate(zip(metrics, data, colors)):
        offset = width * (i - 1.5)
        ax.bar(x + offset, values, width, label=metric.replace(' (%)', ''), color=color)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Model comparison saved to {save_path}")


def visualize_sample_predictions(model, data_loader, class_names, device,
                                 num_samples=16, save_path='../results/predictions.png'):
    """
    Visualize sample predictions with confidence scores.
    
    Args:
        model: Trained model
        data_loader: DataLoader
        class_names: List of class names
        device: 'cuda' or 'cpu'
        num_samples: Number of samples to display
        save_path: Path to save plot
    """
    model.eval()
    
    # Get a batch
    images, labels = next(iter(data_loader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    
    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
    
    # Move to CPU for plotting
    images = images.cpu()
    predictions = predictions.cpu()
    confidences = confidences.cpu()
    
    # Denormalize images for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)
    
    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        img = images[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].axis('off')
        
        true_class = class_names[labels[i]]
        pred_class = class_names[predictions[i]]
        confidence = confidences[i].item() * 100
        
        color = 'green' if labels[i] == predictions[i] else 'red'
        axes[i].set_title(
            f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.1f}%',
            color=color, fontsize=10, fontweight='bold'
        )
    
    plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Sample predictions saved to {save_path}")


def plot_class_distribution(labels, class_names, save_path='../results/class_distribution.png'):
    """
    Plot class distribution as bar chart and pie chart.
    
    Args:
        labels: Array of labels
        class_names: List of class names
        save_path: Path to save plot
    """
    class_counts = np.bincount(labels)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    axes[0].bar(class_names, class_counts, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add count labels
    for i, count in enumerate(class_counts):
        axes[0].text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    axes[1].pie(class_counts, labels=class_names, autopct='%1.1f%%',
               colors=colors, startangle=90)
    axes[1].set_title('Class Distribution (%)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Class distribution saved to {save_path}")
