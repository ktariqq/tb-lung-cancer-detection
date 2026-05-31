"""
Ensemble methods for combining multiple model predictions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict


class EnsembleClassifier:
    """
    Ensemble classifier that combines predictions from multiple models.
    Supports multiple aggregation strategies.
    """
    def __init__(self, models: List[nn.Module], strategy='average', weights=None):
        """
        Args:
            models: List of trained PyTorch models
            strategy: 'average', 'weighted', 'voting', or 'max'
            weights: Optional weights for 'weighted' strategy
        """
        self.models = models
        self.strategy = strategy
        self.weights = weights
        
        if strategy == 'weighted' and weights is None:
            # Default to equal weights
            self.weights = [1.0 / len(models)] * len(models)
        
        # Set all models to evaluation mode
        for model in self.models:
            model.eval()
    
    def predict(self, x, return_confidence=False):
        """
        Make ensemble predictions.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            return_confidence: Whether to return confidence scores
        
        Returns:
            Predicted classes and optionally confidence scores
        """
        with torch.no_grad():
            # Get predictions from all models
            predictions = []
            for model in self.models:
                output = model(x)
                probs = torch.softmax(output, dim=1)
                predictions.append(probs)
            
            # Stack predictions
            predictions = torch.stack(predictions)  # (num_models, batch_size, num_classes)
            
            # Apply ensemble strategy
            if self.strategy == 'average':
                ensemble_probs = predictions.mean(dim=0)
            
            elif self.strategy == 'weighted':
                weights_tensor = torch.tensor(self.weights, device=x.device).view(-1, 1, 1)
                ensemble_probs = (predictions * weights_tensor).sum(dim=0)
            
            elif self.strategy == 'voting':
                # Hard voting: each model votes for one class
                votes = predictions.argmax(dim=2)  # (num_models, batch_size)
                ensemble_preds = torch.mode(votes, dim=0).values
                
                if return_confidence:
                    # Count votes for predicted class
                    batch_size = votes.shape[1]
                    confidence = torch.zeros(batch_size, device=x.device)
                    for i in range(batch_size):
                        vote_counts = torch.bincount(votes[:, i])
                        confidence[i] = vote_counts[ensemble_preds[i]].float() / len(self.models)
                    return ensemble_preds, confidence
                return ensemble_preds
            
            elif self.strategy == 'max':
                # Take maximum probability across models
                ensemble_probs = predictions.max(dim=0).values
            
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
            
            # Get final predictions
            ensemble_preds = ensemble_probs.argmax(dim=1)
            
            if return_confidence:
                # Get confidence for predicted class
                confidence = ensemble_probs.gather(1, ensemble_preds.unsqueeze(1)).squeeze()
                return ensemble_preds, confidence
            
            return ensemble_preds
    
    def predict_proba(self, x):
        """
        Get probability distributions for ensemble predictions.
        
        Args:
            x: Input tensor
        
        Returns:
            Probability distributions (batch_size, num_classes)
        """
        with torch.no_grad():
            predictions = []
            for model in self.models:
                output = model(x)
                probs = torch.softmax(output, dim=1)
                predictions.append(probs)
            
            predictions = torch.stack(predictions)
            
            if self.strategy in ['average', 'weighted']:
                if self.strategy == 'weighted':
                    weights_tensor = torch.tensor(self.weights, device=x.device).view(-1, 1, 1)
                    ensemble_probs = (predictions * weights_tensor).sum(dim=0)
                else:
                    ensemble_probs = predictions.mean(dim=0)
            else:
                ensemble_probs = predictions.mean(dim=0)
            
            return ensemble_probs


def compute_optimal_weights(models, val_loader, device, num_classes):
    """
    Compute optimal weights for ensemble based on validation performance.
    
    Args:
        models: List of models
        val_loader: Validation data loader
        device: 'cuda' or 'cpu'
        num_classes: Number of classes
    
    Returns:
        List of optimal weights
    """
    from sklearn.metrics import f1_score
    
    print("\nComputing optimal ensemble weights...")
    
    # Get validation predictions from each model
    model_scores = []
    
    for i, model in enumerate(models):
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate F1-score
        f1 = f1_score(all_labels, all_preds, average='weighted')
        model_scores.append(f1)
        print(f"  Model {i+1} F1-Score: {f1:.4f}")
    
    # Convert scores to weights (normalize to sum to 1)
    weights = np.array(model_scores)
    weights = weights / weights.sum()
    
    print(f"\nOptimal weights: {weights}")
    return weights.tolist()
