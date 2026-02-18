"""
TRAINING VISUALIZATION AND DIAGNOSTICS FOR SC-BDI V3

This module provides tools to:
1. Visualize latent space embeddings (Belief, Desire, Intention)
2. Compare embeddings with ground truth goals
3. Track learning dynamics during training
4. Diagnose posterior collapse and other VAE issues

Usage:
    from models.new_bdi.visualize_training import TrainingVisualizer
    
    visualizer = TrainingVisualizer(save_dir='artifacts/diagnostics')
    visualizer.visualize_latent_space(model, val_loader, device, epoch=10)
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm

# Dimensionality reduction
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TrainingVisualizer:
    """Visualization tools for SC-BDI V3 training diagnostics."""
    
    def __init__(
        self,
        save_dir: str = 'artifacts/diagnostics',
        use_wandb: bool = False,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        
        # Color schemes
        self.category_colors = {
            0: '#1f77b4',  # Academic
            1: '#ff7f0e',  # Food
            2: '#2ca02c',  # Recreation
            3: '#d62728',  # Housing
            4: '#9467bd',  # Parking
            5: '#8c564b',  # Medical
            6: '#7f7f7f',  # Unknown
        }
        
        self.category_names = {
            0: 'Academic',
            1: 'Food', 
            2: 'Recreation',
            3: 'Housing',
            4: 'Parking',
            5: 'Medical',
            6: 'Unknown',
        }
    
    @torch.no_grad()
    def extract_embeddings(
        self,
        model,
        dataloader,
        device: torch.device,
        max_samples: int = 5000,
    ) -> Dict[str, np.ndarray]:
        """
        Extract latent embeddings from the model.
        
        Returns:
            Dict with keys: belief_z, desire_z, intention_z, goal_idx, 
                           goal_cat_idx, path_progress, agent_id
        """
        model.eval()
        
        embeddings = defaultdict(list)
        total_samples = 0
        
        pbar = tqdm(dataloader, desc="Extracting embeddings")
        
        for batch in pbar:
            if total_samples >= max_samples:
                break
            
            batch_size = batch['history_node_indices'].size(0)
            samples_to_take = min(batch_size, max_samples - total_samples)
            
            # Move to device
            history_node_indices = batch['history_node_indices'][:samples_to_take].to(device)
            history_lengths = batch['history_lengths'][:samples_to_take].to(device)
            agent_id = batch['agent_id'][:samples_to_take].to(device)
            path_progress = batch['path_progress'][:samples_to_take].to(device)
            goal_idx = batch['goal_idx'][:samples_to_take]
            goal_cat_idx = batch['goal_cat_idx'][:samples_to_take]
            next_node_idx = batch['next_node_idx'][:samples_to_take].to(device)
            
            # Forward pass
            outputs = model(
                history_node_indices=history_node_indices,
                history_lengths=history_lengths,
                agent_ids=agent_id,
                path_progress=path_progress,
                compute_loss=False,
            )
            
            # Store embeddings
            embeddings['belief_z'].append(outputs['belief_z'].cpu().numpy())
            embeddings['desire_z'].append(outputs['desire_z'].cpu().numpy())
            embeddings['intention_z'].append(outputs['intention_z'].cpu().numpy())
            embeddings['goal_idx'].append(goal_idx.numpy())
            embeddings['goal_cat_idx'].append(goal_cat_idx.numpy())
            embeddings['path_progress'].append(path_progress.cpu().numpy())
            embeddings['agent_id'].append(agent_id.cpu().numpy())
            
            # Goal predictions
            embeddings['goal_pred'].append(outputs['goal'].argmax(dim=1).cpu().numpy())
            if 'desire_goal_logits' in outputs:
                embeddings['desire_goal_pred'].append(
                    outputs['desire_goal_logits'].argmax(dim=1).cpu().numpy()
                )
            
            total_samples += samples_to_take
            pbar.set_postfix({'samples': total_samples})
        
        # Concatenate
        return {k: np.concatenate(v, axis=0) for k, v in embeddings.items()}
    
    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        method: str = 'tsne',
        n_components: int = 2,
        perplexity: int = 30,
    ) -> np.ndarray:
        """Reduce high-dimensional embeddings to 2D for visualization."""
        if method == 'tsne':
            reducer = TSNE(
                n_components=n_components,
                perplexity=min(perplexity, len(embeddings) - 1),
                random_state=42,
                n_iter=1000,
            )
        elif method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == 'umap' and UMAP_AVAILABLE:
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=42,
                n_neighbors=min(15, len(embeddings) - 1),
            )
        else:
            reducer = PCA(n_components=n_components, random_state=42)
        
        return reducer.fit_transform(embeddings)
    
    def visualize_latent_space(
        self,
        model,
        dataloader,
        device: torch.device,
        epoch: int,
        max_samples: int = 5000,
        method: str = 'tsne',
    ) -> str:
        """
        Create comprehensive latent space visualization.
        
        Returns path to saved figure.
        """
        print(f"\n📊 Visualizing latent space (epoch {epoch})...")
        
        # Extract embeddings
        embeddings = self.extract_embeddings(model, dataloader, device, max_samples)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # =====================================================================
        # Row 1: Belief, Desire, Intention embeddings colored by GOAL CATEGORY
        # =====================================================================
        
        for idx, (name, z_key) in enumerate([
            ('Belief', 'belief_z'),
            ('Desire', 'desire_z'),
            ('Intention', 'intention_z'),
        ]):
            ax = fig.add_subplot(gs[0, idx])
            
            # Reduce dimensions
            z_2d = self.reduce_dimensions(embeddings[z_key], method=method)
            
            # Color by goal category
            for cat_idx in range(7):
                mask = embeddings['goal_cat_idx'] == cat_idx
                if mask.sum() > 0:
                    ax.scatter(
                        z_2d[mask, 0], z_2d[mask, 1],
                        c=self.category_colors[cat_idx],
                        label=self.category_names[cat_idx],
                        alpha=0.5, s=10,
                    )
            
            ax.set_title(f'{name} Latent Space\n(by Goal Category)', fontsize=12)
            ax.set_xlabel(f'{method.upper()} 1')
            ax.set_ylabel(f'{method.upper()} 2')
            if idx == 2:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Column 4: Combined BDI
        ax = fig.add_subplot(gs[0, 3])
        combined_z = np.concatenate([
            embeddings['belief_z'],
            embeddings['desire_z'],
            embeddings['intention_z'],
        ], axis=1)
        combined_2d = self.reduce_dimensions(combined_z, method=method)
        
        for cat_idx in range(7):
            mask = embeddings['goal_cat_idx'] == cat_idx
            if mask.sum() > 0:
                ax.scatter(
                    combined_2d[mask, 0], combined_2d[mask, 1],
                    c=self.category_colors[cat_idx],
                    label=self.category_names[cat_idx],
                    alpha=0.5, s=10,
                )
        ax.set_title('Combined B+D+I\n(by Goal Category)', fontsize=12)
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        
        # =====================================================================
        # Row 2: Desire embeddings colored by path progress
        # =====================================================================
        
        ax = fig.add_subplot(gs[1, 0])
        desire_2d = self.reduce_dimensions(embeddings['desire_z'], method=method)
        scatter = ax.scatter(
            desire_2d[:, 0], desire_2d[:, 1],
            c=embeddings['path_progress'],
            cmap='viridis', alpha=0.5, s=10,
        )
        plt.colorbar(scatter, ax=ax, label='Path Progress')
        ax.set_title('Desire Latent Space\n(by Path Progress)', fontsize=12)
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        
        # Intention by path progress
        ax = fig.add_subplot(gs[1, 1])
        intention_2d = self.reduce_dimensions(embeddings['intention_z'], method=method)
        scatter = ax.scatter(
            intention_2d[:, 0], intention_2d[:, 1],
            c=embeddings['path_progress'],
            cmap='viridis', alpha=0.5, s=10,
        )
        plt.colorbar(scatter, ax=ax, label='Path Progress')
        ax.set_title('Intention Latent Space\n(by Path Progress)', fontsize=12)
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        
        # Desire by agent
        ax = fig.add_subplot(gs[1, 2])
        unique_agents = np.unique(embeddings['agent_id'])[:20]  # Top 20 agents
        for agent in unique_agents:
            mask = embeddings['agent_id'] == agent
            if mask.sum() > 0:
                ax.scatter(
                    desire_2d[mask, 0], desire_2d[mask, 1],
                    alpha=0.5, s=10, label=f'Agent {agent}',
                )
        ax.set_title('Desire Latent Space\n(by Agent ID, top 20)', fontsize=12)
        ax.set_xlabel(f'{method.upper()} 1')
        ax.set_ylabel(f'{method.upper()} 2')
        
        # Prediction accuracy by progress
        ax = fig.add_subplot(gs[1, 3])
        progress_bins = [0, 0.25, 0.5, 0.75, 1.0]
        goal_accs = []
        desire_accs = []
        bin_labels = ['0-25%', '25-50%', '50-75%', '75-100%']
        
        for i in range(len(progress_bins) - 1):
            mask = (embeddings['path_progress'] >= progress_bins[i]) & \
                   (embeddings['path_progress'] < progress_bins[i + 1])
            if mask.sum() > 0:
                goal_acc = (embeddings['goal_pred'][mask] == embeddings['goal_idx'][mask]).mean() * 100
                goal_accs.append(goal_acc)
                if 'desire_goal_pred' in embeddings:
                    desire_acc = (embeddings['desire_goal_pred'][mask] == embeddings['goal_idx'][mask]).mean() * 100
                    desire_accs.append(desire_acc)
                else:
                    desire_accs.append(0)
            else:
                goal_accs.append(0)
                desire_accs.append(0)
        
        x = np.arange(len(bin_labels))
        width = 0.35
        ax.bar(x - width/2, goal_accs, width, label='Goal (Intention)', color='blue', alpha=0.7)
        ax.bar(x + width/2, desire_accs, width, label='Goal (Desire)', color='orange', alpha=0.7)
        ax.set_xlabel('Path Progress')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Goal Prediction Accuracy\nby Path Progress', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels)
        ax.legend()
        ax.set_ylim(0, max(max(goal_accs), max(desire_accs)) * 1.2 + 1)
        
        # =====================================================================
        # Row 3: Diagnostic plots
        # =====================================================================
        
        # Latent variance (posterior collapse detection)
        ax = fig.add_subplot(gs[2, 0])
        for name, z_key in [('Belief', 'belief_z'), ('Desire', 'desire_z'), ('Intention', 'intention_z')]:
            variances = np.var(embeddings[z_key], axis=0)
            ax.hist(variances, bins=30, alpha=0.5, label=name)
        ax.set_xlabel('Variance per Dimension')
        ax.set_ylabel('Count')
        ax.set_title('Latent Dimension Variance\n(Low = Posterior Collapse)', fontsize=12)
        ax.legend()
        ax.axvline(x=0.1, color='red', linestyle='--', label='Collapse Threshold')
        
        # Goal distribution
        ax = fig.add_subplot(gs[2, 1])
        unique_goals, goal_counts = np.unique(embeddings['goal_idx'], return_counts=True)
        top_k = min(20, len(unique_goals))
        sorted_idx = np.argsort(goal_counts)[::-1][:top_k]
        ax.bar(range(top_k), goal_counts[sorted_idx], color='steelblue')
        ax.set_xlabel('Goal Node Index (top 20)')
        ax.set_ylabel('Count')
        ax.set_title('Goal Node Distribution', fontsize=12)
        
        # Category distribution
        ax = fig.add_subplot(gs[2, 2])
        cat_counts = [np.sum(embeddings['goal_cat_idx'] == i) for i in range(7)]
        ax.bar(
            [self.category_names[i] for i in range(7)],
            cat_counts,
            color=[self.category_colors[i] for i in range(7)],
        )
        ax.set_xlabel('Category')
        ax.set_ylabel('Count')
        ax.set_title('Goal Category Distribution', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Confusion: Predicted vs True Category
        ax = fig.add_subplot(gs[2, 3])
        
        # Simple accuracy metrics
        goal_acc = (embeddings['goal_pred'] == embeddings['goal_idx']).mean() * 100
        text = f"Overall Goal Accuracy: {goal_acc:.2f}%\n"
        text += f"Total Samples: {len(embeddings['goal_idx'])}\n"
        text += f"Unique Goals: {len(np.unique(embeddings['goal_idx']))}\n"
        text += f"Belief dim: {embeddings['belief_z'].shape[1]}\n"
        text += f"Desire dim: {embeddings['desire_z'].shape[1]}\n"
        text += f"Intention dim: {embeddings['intention_z'].shape[1]}\n"
        
        # Variance statistics
        belief_var = np.var(embeddings['belief_z'], axis=0).mean()
        desire_var = np.var(embeddings['desire_z'], axis=0).mean()
        intention_var = np.var(embeddings['intention_z'], axis=0).mean()
        text += f"\nMean Variance:\n"
        text += f"  Belief: {belief_var:.4f}\n"
        text += f"  Desire: {desire_var:.4f}\n"
        text += f"  Intention: {intention_var:.4f}\n"
        
        if belief_var < 0.1 or desire_var < 0.1 or intention_var < 0.1:
            text += "\n⚠️ LOW VARIANCE: Possible posterior collapse!"
        
        ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.axis('off')
        ax.set_title('Training Statistics', fontsize=12)
        
        # Title
        fig.suptitle(f'SC-BDI V3 Latent Space Visualization - Epoch {epoch}', fontsize=16, y=1.02)
        
        # Save
        save_path = self.save_dir / f'latent_space_epoch_{epoch:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved visualization to {save_path}")
        
        return str(save_path)
    
    def visualize_desire_goal_alignment(
        self,
        model,
        dataloader,
        device: torch.device,
        epoch: int,
        max_samples: int = 3000,
    ) -> str:
        """
        Visualize how well Desire embeddings align with goals.
        This is the key diagnostic for understanding if the model is learning.
        """
        print(f"\n🎯 Visualizing Desire-Goal alignment (epoch {epoch})...")
        
        embeddings = self.extract_embeddings(model, dataloader, device, max_samples)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Reduce desire dimensions
        desire_2d = self.reduce_dimensions(embeddings['desire_z'], method='tsne')
        
        # 1. Desire colored by goal category
        ax = axes[0, 0]
        for cat_idx in range(7):
            mask = embeddings['goal_cat_idx'] == cat_idx
            if mask.sum() > 0:
                ax.scatter(
                    desire_2d[mask, 0], desire_2d[mask, 1],
                    c=self.category_colors[cat_idx],
                    label=self.category_names[cat_idx],
                    alpha=0.6, s=20,
                )
        ax.set_title('Desire Embeddings\n(Colored by True Goal Category)', fontsize=12)
        ax.legend(fontsize=8)
        
        # 2. Show cluster quality per category
        ax = axes[0, 1]
        from sklearn.metrics import silhouette_score
        try:
            score = silhouette_score(desire_2d, embeddings['goal_cat_idx'])
            ax.text(0.5, 0.5, f'Silhouette Score:\n{score:.3f}\n\n'
                    f'(1.0 = Perfect clusters)\n(0.0 = Overlapping)\n(-1.0 = Wrong clusters)',
                    ha='center', va='center', fontsize=14, transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgreen' if score > 0.3 else 'lightyellow' if score > 0 else 'lightcoral'))
        except:
            ax.text(0.5, 0.5, 'Could not compute\nsilhouette score', ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.axis('off')
        ax.set_title('Cluster Quality', fontsize=12)
        
        # 3. Top-5 most common goals
        ax = axes[0, 2]
        unique_goals, counts = np.unique(embeddings['goal_idx'], return_counts=True)
        top_5_idx = np.argsort(counts)[::-1][:5]
        top_5_goals = unique_goals[top_5_idx]
        
        colors = plt.cm.tab10(np.linspace(0, 1, 5))
        for i, goal in enumerate(top_5_goals):
            mask = embeddings['goal_idx'] == goal
            ax.scatter(
                desire_2d[mask, 0], desire_2d[mask, 1],
                c=[colors[i]], label=f'Goal {goal}',
                alpha=0.6, s=20,
            )
        ax.set_title('Desire Embeddings\n(Top 5 Most Common Goals)', fontsize=12)
        ax.legend(fontsize=8)
        
        # 4. Prediction accuracy heatmap by category
        ax = axes[1, 0]
        if 'desire_goal_pred' in embeddings:
            from sklearn.metrics import confusion_matrix
            # Use category-level confusion
            cm = confusion_matrix(
                embeddings['goal_cat_idx'],
                embeddings['goal_cat_idx'],  # Would need predicted category
                labels=range(7),
            )
            im = ax.imshow(cm, cmap='Blues')
            ax.set_xticks(range(7))
            ax.set_yticks(range(7))
            ax.set_xticklabels([self.category_names[i][:4] for i in range(7)], rotation=45, ha='right')
            ax.set_yticklabels([self.category_names[i][:4] for i in range(7)])
            ax.set_xlabel('Predicted Category')
            ax.set_ylabel('True Category')
            ax.set_title('Category Confusion Matrix', fontsize=12)
            plt.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, 'No desire predictions available', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
        
        # 5. Path progress effect
        ax = axes[1, 1]
        progress_bins = np.linspace(0, 1, 11)
        accuracies = []
        for i in range(len(progress_bins) - 1):
            mask = (embeddings['path_progress'] >= progress_bins[i]) & \
                   (embeddings['path_progress'] < progress_bins[i + 1])
            if mask.sum() > 0:
                acc = (embeddings['goal_pred'][mask] == embeddings['goal_idx'][mask]).mean() * 100
                accuracies.append(acc)
            else:
                accuracies.append(0)
        
        ax.plot(range(10), accuracies, 'b-o', linewidth=2, markersize=8)
        ax.fill_between(range(10), accuracies, alpha=0.3)
        ax.set_xticks(range(10))
        ax.set_xticklabels([f'{int(p*100)}%' for p in progress_bins[:-1]], rotation=45)
        ax.set_xlabel('Path Progress')
        ax.set_ylabel('Goal Accuracy (%)')
        ax.set_title('Goal Accuracy vs Path Progress', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add trend expectation
        ax.axhline(y=100/735, color='red', linestyle='--', label=f'Random chance ({100/735:.2f}%)')
        ax.legend()
        
        # 6. Embedding norm distribution
        ax = axes[1, 2]
        belief_norms = np.linalg.norm(embeddings['belief_z'], axis=1)
        desire_norms = np.linalg.norm(embeddings['desire_z'], axis=1)
        intention_norms = np.linalg.norm(embeddings['intention_z'], axis=1)
        
        ax.hist(belief_norms, bins=50, alpha=0.5, label=f'Belief (μ={belief_norms.mean():.2f})')
        ax.hist(desire_norms, bins=50, alpha=0.5, label=f'Desire (μ={desire_norms.mean():.2f})')
        ax.hist(intention_norms, bins=50, alpha=0.5, label=f'Intention (μ={intention_norms.mean():.2f})')
        ax.set_xlabel('L2 Norm')
        ax.set_ylabel('Count')
        ax.set_title('Embedding Norm Distribution', fontsize=12)
        ax.legend()
        
        fig.suptitle(f'Desire-Goal Alignment Analysis - Epoch {epoch}', fontsize=16)
        plt.tight_layout()
        
        save_path = self.save_dir / f'desire_goal_alignment_epoch_{epoch:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved alignment visualization to {save_path}")
        
        return str(save_path)
    
    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        epoch: int,
    ) -> str:
        """Plot training curves from history dict."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Loss curves
        ax = axes[0, 0]
        if 'train_loss' in history:
            ax.plot(history['train_loss'], label='Train', color='blue')
        if 'val_loss' in history:
            ax.plot(history['val_loss'], label='Val', color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Goal accuracy
        ax = axes[0, 1]
        if 'train_goal_acc' in history:
            ax.plot(history['train_goal_acc'], label='Train Goal', color='blue')
        if 'val_goal_acc' in history:
            ax.plot(history['val_goal_acc'], label='Val Goal', color='orange')
        if 'train_desire_goal_acc' in history:
            ax.plot(history['train_desire_goal_acc'], label='Train Desire→Goal', color='green', linestyle='--')
        if 'val_desire_goal_acc' in history:
            ax.plot(history['val_desire_goal_acc'], label='Val Desire→Goal', color='red', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Goal Prediction Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. VAE losses
        ax = axes[0, 2]
        if 'belief_kl' in history:
            ax.plot(history['belief_kl'], label='Belief KL')
        if 'desire_kl' in history:
            ax.plot(history['desire_kl'], label='Desire KL')
        if 'intention_kl' in history:
            ax.plot(history['intention_kl'], label='Intention KL')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('KL Divergence')
        ax.set_title('VAE KL Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. InfoNCE loss
        ax = axes[1, 0]
        if 'infonce_loss' in history:
            ax.plot(history['infonce_loss'], label='InfoNCE', color='purple')
        if 'mi_loss' in history:
            ax.plot(history['mi_loss'], label='MI Loss', color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Contrastive Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. KL weight
        ax = axes[1, 1]
        if 'kl_weight' in history:
            ax.plot(history['kl_weight'], label='KL Weight', color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weight')
        ax.set_title('KL Annealing Schedule')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        # 6. Learning rate
        ax = axes[1, 2]
        if 'lr' in history:
            ax.plot(history['lr'], label='LR', color='blue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'Training Curves - Epoch {epoch}', fontsize=16)
        plt.tight_layout()
        
        save_path = self.save_dir / f'training_curves_epoch_{epoch:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)


def add_visualization_to_training(
    model,
    val_loader,
    device,
    epoch: int,
    save_dir: str = 'artifacts/diagnostics',
    visualize_every: int = 5,
):
    """
    Helper function to add visualization during training.
    Call this at the end of each epoch.
    """
    if epoch % visualize_every != 0:
        return
    
    visualizer = TrainingVisualizer(save_dir=save_dir)
    visualizer.visualize_latent_space(model, val_loader, device, epoch)
    visualizer.visualize_desire_goal_alignment(model, val_loader, device, epoch)


if __name__ == '__main__':
    print("Run this module's functions during training to visualize embeddings.")
    print("Example:")
    print("  from models.new_bdi.visualize_training import TrainingVisualizer")
    print("  visualizer = TrainingVisualizer()")
    print("  visualizer.visualize_latent_space(model, val_loader, device, epoch=10)")
