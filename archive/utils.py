import torch
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional

def get_device() -> torch.device:
    """
    Get the best available device for training.
    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon GPU) > CPU
    
    Returns:
        torch.device: The selected device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}") # type: ignore
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"🚀 Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print(f"⚠️  Using CPU (training will be slower)")
    
    return device


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict,
    filepath: str
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint saved: {filepath}")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Tuple[int, float, Dict]:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
    
    Returns:
        Tuple of (epoch, loss, metrics)
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    metrics = checkpoint.get('metrics', {})
    
    print(f"📥 Checkpoint loaded: {filepath}")
    print(f"   Epoch: {epoch}, Loss: {loss:.4f}")
    
    return epoch, loss, metrics


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    """
    Compute top-k accuracy.
    
    Args:
        predictions: (batch_size, num_classes) - Model predictions (logits or probs)
        targets: (batch_size,) - True class indices
        k: Number of top predictions to consider
    
    Returns:
        float: Top-k accuracy as percentage
    """
    _, top_k_preds = torch.topk(predictions, k, dim=-1)
    targets_expanded = targets.unsqueeze(-1).expand_as(top_k_preds)
    correct = (top_k_preds == targets_expanded).any(dim=-1).float()
    accuracy = correct.mean().item() * 100
    return accuracy


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.metrics = {}  # For storing multiple metrics
    
    def update(self, val, n=1):
        """Update with single value or dict of values."""
        if isinstance(val, dict):
            # Update multiple metrics
            for key, value in val.items():
                if key not in self.metrics:
                    self.metrics[key] = {'sum': 0, 'count': 0}
                self.metrics[key]['sum'] += value * n
                self.metrics[key]['count'] += n
        else:
            # Update single value
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count if self.count > 0 else 0
    
    def avg(self, key: str = None) -> float:
        """Get average of a specific metric or overall average."""
        if key is not None:
            if key in self.metrics:
                return self.metrics[key]['sum'] / max(self.metrics[key]['count'], 1)
            return 0.0
        return self.avg
    
    def get_averages(self) -> Dict:
        """Get all metric averages as dict."""
        result = {}
        for key, metric in self.metrics.items():
            result[key] = metric['sum'] / max(metric['count'], 1)
        return result


class MetricsTracker:
    """Track training and validation metrics over time."""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_top5_accs = []
        self.val_top5_accs = []
    
    def update(
        self,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        train_top5_acc: float,
        val_top5_acc: float
    ):
        """Update metrics for current epoch."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.train_top5_accs.append(train_top5_acc)
        self.val_top5_accs.append(val_top5_acc)
    
    def save(self, filepath: str):
        """Save metrics to JSON file."""
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'train_top5_accs': self.train_top5_accs,
            'val_top5_accs': self.val_top5_accs
        }
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"📊 Metrics saved: {filepath}")
    
    def load(self, filepath: str):
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            metrics = json.load(f)
        self.train_losses = metrics['train_losses']
        self.val_losses = metrics['val_losses']
        self.train_accs = metrics['train_accs']
        self.val_accs = metrics['val_accs']
        self.train_top5_accs = metrics.get('train_top5_accs', [])
        self.val_top5_accs = metrics.get('val_top5_accs', [])
        print(f"📊 Metrics loaded: {filepath}")
    
    def get_best_epoch(self, metric='val_acc') -> Tuple[int, float]:
        """
        Get epoch with best metric value.
        
        Args:
            metric: One of 'val_acc', 'val_top5_acc', 'val_loss'
        
        Returns:
            Tuple of (best_epoch, best_value)
        """
        if metric == 'val_acc':
            values = self.val_accs
            best_idx = np.argmax(values)
        elif metric == 'val_top5_acc':
            values = self.val_top5_accs
            best_idx = np.argmax(values)
        elif metric == 'val_loss':
            values = self.val_losses
            best_idx = np.argmin(values)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return best_idx, values[best_idx] # type: ignore
    
    def print_summary(self):
        """Print training summary."""
        if not self.train_losses:
            print("No metrics to display")
            return
        
        print("\n" + "=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)
        
        best_val_epoch, best_val_acc = self.get_best_epoch('val_acc')
        best_val5_epoch, best_val5_acc = self.get_best_epoch('val_top5_acc')
        best_loss_epoch, best_loss = self.get_best_epoch('val_loss')
        
        print(f"\nBest Validation Accuracy:")
        print(f"  Epoch {best_val_epoch + 1}: {best_val_acc:.2f}%")
        print(f"\nBest Validation Top-5 Accuracy:")
        print(f"  Epoch {best_val5_epoch + 1}: {best_val5_acc:.2f}%")
        print(f"\nBest Validation Loss:")
        print(f"  Epoch {best_loss_epoch + 1}: {best_loss:.4f}")
        
        print(f"\nFinal Epoch ({len(self.train_losses)}):")
        print(f"  Train Loss: {self.train_losses[-1]:.4f} | Val Loss: {self.val_losses[-1]:.4f}")
        print(f"  Train Acc:  {self.train_accs[-1]:.2f}% | Val Acc:  {self.val_accs[-1]:.2f}%")
        print(f"  Train Top-5: {self.train_top5_accs[-1]:.2f}% | Val Top-5: {self.val_top5_accs[-1]:.2f}%")
        print("=" * 80)
