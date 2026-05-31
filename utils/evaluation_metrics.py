"""
Comprehensive evaluation metrics for multi-class medical image classification.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize


def evaluate_model(model, data_loader, device, num_classes):
    """
    Comprehensive model evaluation with all metrics.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for evaluation
        device: 'cuda' or 'cpu'
        num_classes: Number of classes
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Multi-class ROC-AUC (One-vs-Rest)
    try:
        y_binary = label_binarize(all_labels, classes=range(num_classes))
        if num_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc_score = roc_auc_score(y_binary, all_probs, average='weighted', multi_class='ovr')
    except:
        auc_score = None
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'auc_score': auc_score,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def print_evaluation_report(results, class_names):
    """
    Print formatted evaluation report.
    
    Args:
        results: Dictionary from evaluate_model()
        class_names: List of class names
    """
    print("\n" + "="*70)
    print("EVALUATION REPORT")
    print("="*70)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {results['accuracy']*100:.2f}%")
    print(f"  Precision: {results['precision']*100:.2f}%")
    print(f"  Recall:    {results['recall']*100:.2f}%")
    print(f"  F1-Score:  {results['f1_score']*100:.2f}%")
    if results['auc_score'] is not None:
        print(f"  AUC-ROC:   {results['auc_score']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    for i, class_name in enumerate(class_names):
        print(f"\n  {class_name}:")
        print(f"    Precision: {results['precision_per_class'][i]*100:.2f}%")
        print(f"    Recall:    {results['recall_per_class'][i]*100:.2f}%")
        print(f"    F1-Score:  {results['f1_per_class'][i]*100:.2f}%")
    
    print("\n" + "="*70 + "\n")


def create_metrics_dataframe(results_dict, model_names, class_names):
    """
    Create comparison DataFrame for multiple models.
    
    Args:
        results_dict: Dictionary mapping model names to evaluation results
        model_names: List of model names
        class_names: List of class names
    
    Returns:
        pandas DataFrame with comparison
    """
    data = []
    
    for model_name in model_names:
        results = results_dict[model_name]
        
        row = {
            'Model': model_name,
            'Accuracy (%)': f"{results['accuracy']*100:.2f}",
            'Precision (%)': f"{results['precision']*100:.2f}",
            'Recall (%)': f"{results['recall']*100:.2f}",
            'F1-Score (%)': f"{results['f1_score']*100:.2f}"
        }
        
        if results['auc_score'] is not None:
            row['AUC-ROC'] = f"{results['auc_score']:.4f}"
        
        # Add per-class F1 scores
        for i, class_name in enumerate(class_names):
            row[f'{class_name} F1 (%)'] = f"{results['f1_per_class'][i]*100:.2f}"
        
        data.append(row)
    
    return pd.DataFrame(data)


def calculate_confidence_intervals(predictions, labels, n_bootstrap=1000, confidence=0.95):
    """
    Calculate confidence intervals for accuracy using bootstrap method.
    
    Args:
        predictions: Array of predictions
        labels: Array of true labels
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95%)
    
    Returns:
        Dictionary with mean, lower, and upper bounds
    """
    np.random.seed(42)
    
    accuracies = []
    n_samples = len(predictions)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        sample_preds = predictions[indices]
        sample_labels = labels[indices]
        
        # Calculate accuracy
        accuracy = accuracy_score(sample_labels, sample_preds)
        accuracies.append(accuracy)
    
    accuracies = np.array(accuracies)
    
    # Calculate percentiles
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(accuracies, lower_percentile)
    upper_bound = np.percentile(accuracies, upper_percentile)
    mean_accuracy = np.mean(accuracies)
    
    return {
        'mean': mean_accuracy,
        'lower': lower_bound,
        'upper': upper_bound,
        'confidence': confidence
    }


def compute_sensitivity_specificity(cm, class_idx):
    """
    Compute sensitivity and specificity for a specific class (one-vs-rest).
    
    Args:
        cm: Confusion matrix
        class_idx: Index of the class
    
    Returns:
        sensitivity, specificity
    """
    # True positives: diagonal element for this class
    tp = cm[class_idx, class_idx]
    
    # False negatives: sum of row excluding diagonal
    fn = cm[class_idx, :].sum() - tp
    
    # False positives: sum of column excluding diagonal
    fp = cm[:, class_idx].sum() - tp
    
    # True negatives: sum of all other elements
    tn = cm.sum() - tp - fn - fp
    
    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return sensitivity, specificity


def generate_classification_report_df(labels, predictions, class_names):
    """
    Generate classification report as DataFrame.
    
    Args:
        labels: True labels
        predictions: Predicted labels
        class_names: List of class names
    
    Returns:
        pandas DataFrame
    """
    report = classification_report(
        labels, predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(report).transpose()
    
    # Format percentages
    for col in ['precision', 'recall', 'f1-score']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x*100:.2f}%" if isinstance(x, float) else x)
    
    return df
