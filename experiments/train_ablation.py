"""
Ablation Study Training Script for SC-BDI-VAE

Trains all ablation variants sequentially so the full study can run
unattended.  Each variant is saved under ``checkpoints/ablation/<variant>/``
and logged to Weights & Biases with a descriptive run name.

Ablation variants (see bdi_vae_v3_model.py for details):
  full                 – Full SC-BDI-VAE (control)
  no_belief            – Remove BeliefVAE
  no_desire            – Remove DesireVAE
  flat                 – Single monolithic VAE (no BDI decomposition)
  no_conditional_prior – Standard N(0,I) prior for intention
  no_infonce           – Disable InfoNCE contrastive loss
  no_mi                – Disable MI minimization

Resume support:
  • Completed variants are detected by ``best_model.pt`` in their
    checkpoint folder and skipped automatically.
  • A crashed variant can be resumed from its latest epoch checkpoint
    with ``--resume``.
  • To restart a specific variant from scratch, delete its checkpoint
    folder or use ``--restart <variant>``.

Usage:
    # Train all variants (skipping any already completed):
    python experiments/train_ablation.py

    # Resume a crashed run (picks up the unfinished variant):
    python experiments/train_ablation.py --resume

    # Train only specific variants:
    python experiments/train_ablation.py --variants no_belief flat

    # Override epochs / batch size:
    python experiments/train_ablation.py --num_epochs 40 --batch_size 512

    # Skip W&B logging:
    python experiments/train_ablation.py --no_wandb
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# MPS fallback for Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vae_bdi_simple.bdi_vae_v3_model import (
    SequentialConditionalBDIVAE,
    create_sc_bdi_vae_v3,
)
from models.vae_bdi_simple.bdi_dataset_v2 import (
    BDIVAEDatasetV2,
    collate_bdi_samples_v2,
)
from models.utils.utils import get_device, set_seed, save_checkpoint, AverageMeter

# W&B (optional)
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ============================================================================
# ABLATION REGISTRY
# ============================================================================

# Ordered list – controls training sequence
ABLATION_VARIANTS: List[Optional[str]] = [
    None,                   # full model (control)
    "no_belief",
    "no_desire",
    "flat",
    "no_conditional_prior",
    "no_infonce",
    "no_mi",
]

VARIANT_DISPLAY_NAMES: Dict[Optional[str], str] = {
    None:                   "full",
    "no_belief":            "no_belief",
    "no_desire":            "no_desire",
    "flat":                 "flat",
    "no_conditional_prior": "no_conditional_prior",
    "no_infonce":           "no_infonce",
    "no_mi":                "no_mi",
}

VARIANT_WANDB_NAMES: Dict[Optional[str], str] = {
    None:                   "ablation-full",
    "no_belief":            "ablation-no-belief",
    "no_desire":            "ablation-no-desire",
    "flat":                 "ablation-flat",
    "no_conditional_prior": "ablation-no-cond-prior",
    "no_infonce":           "ablation-no-infonce",
    "no_mi":                "ablation-no-mi",
}


# ============================================================================
# KL ANNEALING (reused from train_bdi_vae_v3.py)
# ============================================================================

class KLAnnealingSchedule:
    def __init__(
        self,
        strategy: str = "monotonic",
        warmup_epochs: int = 10,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ):
        self.strategy = strategy
        self.warmup_epochs = warmup_epochs
        self.min_weight = min_weight
        self.max_weight = max_weight

    def get_weight(self, epoch: int) -> float:
        if self.strategy == "monotonic":
            if epoch < self.warmup_epochs:
                return self.min_weight + (self.max_weight - self.min_weight) * (
                    epoch / self.warmup_epochs
                )
            return self.max_weight
        return self.max_weight


# ============================================================================
# DATA LOADING (same logic as train_bdi_vae_v3.py)
# ============================================================================

def load_data(
    data_dir: str,
    graph_path: str,
    split_indices_path: str,
    trajectory_filename: str = "all_trajectories.json",
) -> Tuple:
    """Load graph, trajectories, POI nodes, split indices, and num_agents."""
    import networkx as nx
    from graph_controller.world_graph import WorldGraph

    print(f"\n{'=' * 80}")
    print("📂 LOADING DATA")
    print(f"{'=' * 80}")

    # Graph
    graph = nx.read_graphml(graph_path)
    print(f"  📊 Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Trajectories
    traj_path = Path(data_dir) / trajectory_filename
    with open(traj_path, "r") as f:
        traj_data = json.load(f)

    if isinstance(traj_data, dict):
        trajectories = []
        sorted_agents = sorted(traj_data.keys())
        for agent_idx, agent_key in enumerate(sorted_agents):
            for traj in traj_data[agent_key]:
                traj["agent_id"] = agent_idx
                trajectories.append(traj)
        num_agents = len(sorted_agents)
    elif isinstance(traj_data, list):
        trajectories = traj_data
        agents_path = Path(data_dir).parent / "agents" / "all_agents.json"
        if agents_path.exists():
            with open(agents_path, "r") as f:
                num_agents = len(json.load(f))
        else:
            num_agents = max(1, len(trajectories) // 1000)
        trajs_per_agent = len(trajectories) // num_agents
        for idx, traj in enumerate(trajectories):
            if "agent_id" not in traj:
                traj["agent_id"] = idx // trajs_per_agent
    else:
        raise ValueError(f"Unexpected trajectory data type: {type(traj_data)}")

    print(f"  📊 Loaded {len(trajectories)} trajectories from {num_agents} agents")

    # POI nodes
    world_graph = WorldGraph(graph)
    poi_nodes = world_graph.poi_nodes
    print(f"  📍 {len(poi_nodes)} POI nodes")

    # Splits
    with open(split_indices_path, "r") as f:
        splits = json.load(f)

    train_idx = splits["train_indices"]
    val_idx = splits["val_indices"]

    # Filter invalid indices
    n = len(trajectories)
    train_idx = [i for i in train_idx if i < n]
    val_idx = [i for i in val_idx if i < n]

    print(f"  📊 Splits: {len(train_idx)} train, {len(val_idx)} val")

    return graph, trajectories, poi_nodes, train_idx, val_idx, num_agents


# ============================================================================
# SINGLE-EPOCH TRAIN / VALIDATE
# ============================================================================

def train_one_epoch(
    model: SequentialConditionalBDIVAE,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: Dict[str, nn.Module],
    device: torch.device,
    epoch: int,
    kl_weight: float,
    config: argparse.Namespace,
    scheduler=None,
) -> Dict[str, float]:
    model.train()
    model.set_kl_weight(kl_weight)
    meters: Dict[str, AverageMeter] = defaultdict(AverageMeter)

    pbar = tqdm(loader, desc=f"  Epoch {epoch+1} [Train]", leave=False)
    for batch in pbar:
        bs = batch["history_node_indices"].size(0)

        hist = batch["history_node_indices"].to(device)
        lengths = batch["history_lengths"].to(device)
        next_node = batch["next_node_idx"].to(device)
        goal = batch["goal_idx"].to(device)
        goal_cat = batch["goal_cat_idx"].to(device)
        agent = batch["agent_id"].to(device)
        progress = batch["path_progress"].to(device)

        outputs = model(
            history_node_indices=hist,
            history_lengths=lengths,
            agent_ids=agent,
            path_progress=progress,
            compute_loss=True,
            next_node_idx=next_node,
            goal_idx=goal,
            goal_cat_idx=goal_cat,
        )

        # Skip NaN batches
        if any(torch.isnan(outputs.get(k, torch.tensor(0.0))).any()
               for k in ["belief_z", "desire_z", "intention_z"]):
            continue

        pred_loss = (
            config.goal_weight * criterion["goal"](outputs["goal"], goal)
            + config.nextstep_weight * criterion["nextstep"](outputs["nextstep"], next_node)
            + config.category_weight * criterion["category"](outputs["category"], goal_cat)
        )
        vae_loss = outputs["total_vae_loss"]
        progress_loss = torch.tensor(0.0, device=device)
        if model.use_progress and "progress_loss" in outputs:
            progress_loss = config.progress_weight * outputs["progress_loss"]

        total_loss = pred_loss + config.vae_weight * vae_loss + progress_loss

        if not torch.isfinite(total_loss):
            continue

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Metrics
        goal_acc = (outputs["goal"].argmax(1) == goal).float().mean().item() * 100
        nextstep_acc = (outputs["nextstep"].argmax(1) == next_node).float().mean().item() * 100
        cat_acc = (outputs["category"].argmax(1) == goal_cat).float().mean().item() * 100
        desire_goal_acc = 0.0
        if "desire_goal_logits" in outputs:
            desire_goal_acc = (outputs["desire_goal_logits"].argmax(1) == goal).float().mean().item() * 100

        meters["loss"].update(total_loss.item(), bs)
        meters["pred_loss"].update(pred_loss.item(), bs)
        meters["vae_loss"].update(vae_loss.item(), bs)
        meters["goal_acc"].update(goal_acc, bs)
        meters["nextstep_acc"].update(nextstep_acc, bs)
        meters["category_acc"].update(cat_acc, bs)
        meters["desire_goal_acc"].update(desire_goal_acc, bs)
        meters["belief_loss"].update(outputs["belief_loss"].item(), bs)
        meters["belief_kl"].update(outputs["belief_kl"].item(), bs)
        meters["desire_loss"].update(outputs["desire_loss"].item(), bs)
        meters["desire_kl"].update(outputs["desire_kl"].item(), bs)
        meters["intention_loss"].update(outputs["intention_loss"].item(), bs)
        meters["intention_kl"].update(outputs["intention_kl"].item(), bs)
        meters["mi_loss"].update(outputs["mi_loss"].item(), bs)
        meters["infonce_loss"].update(outputs["infonce_loss"].item(), bs)

        pbar.set_postfix(
            loss=f"{meters['loss'].avg:.4f}",
            goal=f"{meters['goal_acc'].avg:.1f}%",
        )

    return {k: v.avg for k, v in meters.items()}


@torch.no_grad()
def validate(
    model: SequentialConditionalBDIVAE,
    loader: DataLoader,
    criterion: Dict[str, nn.Module],
    device: torch.device,
    config: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    meters: Dict[str, AverageMeter] = defaultdict(AverageMeter)

    for batch in tqdm(loader, desc="  Validating", leave=False):
        bs = batch["history_node_indices"].size(0)

        hist = batch["history_node_indices"].to(device)
        lengths = batch["history_lengths"].to(device)
        next_node = batch["next_node_idx"].to(device)
        goal = batch["goal_idx"].to(device)
        goal_cat = batch["goal_cat_idx"].to(device)
        agent = batch["agent_id"].to(device)
        progress = batch["path_progress"].to(device)

        outputs = model(
            history_node_indices=hist,
            history_lengths=lengths,
            agent_ids=agent,
            path_progress=progress,
            compute_loss=True,
            next_node_idx=next_node,
            goal_idx=goal,
            goal_cat_idx=goal_cat,
        )

        pred_loss = (
            config.goal_weight * criterion["goal"](outputs["goal"], goal)
            + config.nextstep_weight * criterion["nextstep"](outputs["nextstep"], next_node)
            + config.category_weight * criterion["category"](outputs["category"], goal_cat)
        )
        vae_loss = outputs["total_vae_loss"]
        total_loss = pred_loss + config.vae_weight * vae_loss

        goal_acc = (outputs["goal"].argmax(1) == goal).float().mean().item() * 100
        nextstep_acc = (outputs["nextstep"].argmax(1) == next_node).float().mean().item() * 100
        cat_acc = (outputs["category"].argmax(1) == goal_cat).float().mean().item() * 100
        desire_goal_acc = 0.0
        if "desire_goal_logits" in outputs:
            desire_goal_acc = (outputs["desire_goal_logits"].argmax(1) == goal).float().mean().item() * 100

        meters["loss"].update(total_loss.item(), bs)
        meters["goal_acc"].update(goal_acc, bs)
        meters["nextstep_acc"].update(nextstep_acc, bs)
        meters["category_acc"].update(cat_acc, bs)
        meters["desire_goal_acc"].update(desire_goal_acc, bs)
        meters["infonce_loss"].update(outputs["infonce_loss"].item(), bs)
        meters["mi_loss"].update(outputs["mi_loss"].item(), bs)

    return {k: v.avg for k, v in meters.items()}


# ============================================================================
# CHECKPOINT HELPERS
# ============================================================================

def variant_checkpoint_dir(base_dir: str, variant: Optional[str]) -> Path:
    name = VARIANT_DISPLAY_NAMES.get(variant, variant or "full")
    return Path(base_dir) / name


def find_latest_epoch_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """Return the latest ``checkpoint_epoch_*.pt`` file, or None."""
    candidates = sorted(
        ckpt_dir.glob("checkpoint_epoch_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    return candidates[-1] if candidates else None


def is_variant_complete(ckpt_dir: Path) -> bool:
    return (ckpt_dir / "best_model.pt").exists() and (ckpt_dir / "DONE").exists()


# ============================================================================
# TRAIN ONE VARIANT
# ============================================================================

def train_variant(
    ablation_mode: Optional[str],
    config: argparse.Namespace,
    graph,
    trajectories: List,
    poi_nodes: List,
    train_idx: List[int],
    val_idx: List[int],
    num_agents: int,
    device: torch.device,
) -> None:
    """Train a single ablation variant end-to-end."""

    display = VARIANT_DISPLAY_NAMES[ablation_mode]
    ckpt_dir = variant_checkpoint_dir(config.checkpoint_base, ablation_mode)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Already done? -------------------------------------------------
    if is_variant_complete(ckpt_dir):
        print(f"\n⏭️  Skipping '{display}' — already complete (DONE marker found)")
        return

    print(f"\n{'=' * 80}")
    print(f"🔬 TRAINING ABLATION VARIANT: {display}")
    print(f"   ablation_mode = {ablation_mode!r}")
    print(f"   checkpoint dir = {ckpt_dir}")
    print(f"{'=' * 80}")

    # --- Datasets (created once, reused across variants) ---------------
    node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}
    num_nodes = graph.number_of_nodes()
    num_poi_nodes = len(poi_nodes)

    train_trajs = [trajectories[i] for i in train_idx]
    val_trajs = [trajectories[i] for i in val_idx]

    train_dataset = BDIVAEDatasetV2(
        trajectories=train_trajs,
        graph=graph,
        poi_nodes=poi_nodes,
        node_to_idx_map=node_to_idx,
        include_progress=True,
        include_temporal=True,
    )
    val_dataset = BDIVAEDatasetV2(
        trajectories=val_trajs,
        graph=graph,
        poi_nodes=poi_nodes,
        node_to_idx_map=node_to_idx,
        include_progress=True,
        include_temporal=True,
    )

    num_categories = len(train_dataset.CATEGORY_TO_IDX)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_bdi_samples_v2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_bdi_samples_v2,
        pin_memory=True,
    )

    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Val samples:   {len(val_dataset):,}")
    print(f"  Nodes: {num_nodes}  POIs: {num_poi_nodes}  Categories: {num_categories}  Agents: {num_agents}")

    # --- Create model --------------------------------------------------
    model = create_sc_bdi_vae_v3(
        num_nodes=num_nodes,
        num_agents=num_agents,
        num_poi_nodes=num_poi_nodes,
        num_categories=num_categories,
        node_embedding_dim=config.node_embedding_dim,
        fusion_dim=config.fusion_dim,
        belief_latent_dim=config.belief_latent_dim,
        desire_latent_dim=config.desire_latent_dim,
        intention_latent_dim=config.intention_latent_dim,
        vae_hidden_dim=config.vae_hidden_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        beta_belief=config.beta_belief,
        beta_desire=config.beta_desire,
        beta_intention=config.beta_intention,
        mi_weight=config.mi_weight,
        infonce_weight=config.infonce_weight,
        desire_goal_weight=config.desire_goal_weight,
        free_bits=config.free_bits,
        use_progress=True,
        ablation_mode=ablation_mode,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    steps_per_epoch = len(train_loader)
    total_steps = config.num_epochs * steps_per_epoch

    kl_scheduler = KLAnnealingSchedule(
        strategy="monotonic", warmup_epochs=config.kl_warmup_epochs
    )
    criterion = {
        "goal": nn.CrossEntropyLoss(),
        "nextstep": nn.CrossEntropyLoss(),
        "category": nn.CrossEntropyLoss(),
    }

    # --- Resume from checkpoint? ----------------------------------------
    start_epoch = 0
    best_val_acc = 0.0

    if config.resume:
        latest = find_latest_epoch_checkpoint(ckpt_dir)
        if latest is not None:
            ckpt = torch.load(latest, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_acc = ckpt.get("val_metric", 0.0)
            print(f"  ♻️  Resumed from {latest.name} (epoch {start_epoch}, best_val_acc={best_val_acc:.2f}%)")
        else:
            print("  ℹ️  No checkpoint found to resume — starting fresh")

    # Rebuild scheduler with correct starting step
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    # Fast-forward scheduler to the right step if resuming
    if start_epoch > 0:
        for _ in range(start_epoch * steps_per_epoch):
            scheduler.step()

    # --- W&B -----------------------------------------------------------
    wandb_run = None
    if WANDB_AVAILABLE and config.use_wandb:
        wandb_run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=VARIANT_WANDB_NAMES[ablation_mode],
            group="ablation-study",
            tags=["ablation", display],
            config={
                "ablation_mode": display,
                "num_epochs": config.num_epochs,
                "batch_size": config.batch_size,
                "lr": config.lr,
                "total_params": total_params,
                **{k: v for k, v in vars(config).items()
                   if k not in ("resume", "variants", "restart")},
            },
            reinit=True,
        )

    # --- Training loop --------------------------------------------------
    t0 = time.time()

    for epoch in range(start_epoch, config.num_epochs):
        kl_weight = kl_scheduler.get_weight(epoch)

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, kl_weight, config, scheduler,
        )
        val_metrics = validate(model, val_loader, criterion, device, config)

        lr = optimizer.param_groups[0]["lr"]

        # Console
        print(
            f"  [{display}] Epoch {epoch+1}/{config.num_epochs}  "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_goal={val_metrics['goal_acc']:.2f}%  "
            f"val_cat={val_metrics['category_acc']:.2f}%  "
            f"kl_w={kl_weight:.2f}  lr={lr:.2e}"
        )

        # W&B
        if wandb_run is not None:
            log = {"epoch": epoch + 1, "kl_weight": kl_weight, "lr": lr}
            for k, v in train_metrics.items():
                log[f"train/{k}"] = v
            for k, v in val_metrics.items():
                log[f"val/{k}"] = v
            wandb.log(log)

        # Save best
        if val_metrics["goal_acc"] > best_val_acc:
            best_val_acc = val_metrics["goal_acc"]
            save_checkpoint(
                filepath=str(ckpt_dir / "best_model.pt"),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_metric=best_val_acc,
                is_best=True,
                config={"ablation_mode": display, **vars(config)},
            )
            print(f"    🎯 New best! val_goal_acc = {best_val_acc:.2f}%")

        # Periodic checkpoint (for resume)
        if (epoch + 1) % config.save_every == 0 or epoch == config.num_epochs - 1:
            save_checkpoint(
                filepath=str(ckpt_dir / f"checkpoint_epoch_{epoch+1}.pt"),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_metric=val_metrics["goal_acc"],
                is_best=False,
                config={"ablation_mode": display, **vars(config)},
            )

    # Mark variant as complete
    elapsed = (time.time() - t0) / 60
    (ckpt_dir / "DONE").write_text(
        f"Completed {config.num_epochs} epochs in {elapsed:.1f} min. "
        f"Best val goal acc: {best_val_acc:.2f}%\n"
    )
    print(f"  ✅ '{display}' done in {elapsed:.1f} min  (best val_goal_acc={best_val_acc:.2f}%)")

    # Clean up epoch checkpoints to save disk (keep best_model.pt)
    if not config.keep_epoch_checkpoints:
        for ckpt_file in ckpt_dir.glob("checkpoint_epoch_*.pt"):
            ckpt_file.unlink()
            print(f"    🗑️  Removed {ckpt_file.name}")

    if wandb_run is not None:
        wandb.finish()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ablation Study: train all SC-BDI-VAE variants sequentially",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Variant selection ---
    parser.add_argument(
        "--variants", nargs="*", default=None,
        help="Train only these variants (e.g. --variants no_belief flat). "
             "Use 'full' for the control model. Default: all.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume the last incomplete variant from its latest epoch checkpoint.",
    )
    parser.add_argument(
        "--restart", nargs="*", default=None,
        help="Delete checkpoints for these variants and retrain from scratch.",
    )

    # --- Data ---
    parser.add_argument("--data_dir", type=str,
                        default="data/simulation_data/run_8/trajectories")
    parser.add_argument("--trajectory_filename", type=str,
                        default="all_trajectories.json")
    parser.add_argument("--graph_path", type=str,
                        default="data/processed/ucsd_walk_full.graphml")
    parser.add_argument("--split_indices_path", type=str,
                        default="data/simulation_data/run_8/split_data/split_indices_seed42.json")

    # --- Architecture (match training defaults) ---
    parser.add_argument("--node_embedding_dim", type=int, default=64)
    parser.add_argument("--fusion_dim", type=int, default=128)
    parser.add_argument("--belief_latent_dim", type=int, default=32)
    parser.add_argument("--desire_latent_dim", type=int, default=16)
    parser.add_argument("--intention_latent_dim", type=int, default=32)
    parser.add_argument("--vae_hidden_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    # --- VAE loss weights ---
    parser.add_argument("--beta_belief", type=float, default=1.0)
    parser.add_argument("--beta_desire", type=float, default=1.0)
    parser.add_argument("--beta_intention", type=float, default=1.0)
    parser.add_argument("--mi_weight", type=float, default=0.1)
    parser.add_argument("--infonce_weight", type=float, default=1.0)
    parser.add_argument("--desire_goal_weight", type=float, default=0.5)
    parser.add_argument("--free_bits", type=float, default=0.5)

    # --- Prediction loss weights ---
    parser.add_argument("--goal_weight", type=float, default=1.0)
    parser.add_argument("--nextstep_weight", type=float, default=0.5)
    parser.add_argument("--category_weight", type=float, default=0.3)
    parser.add_argument("--vae_weight", type=float, default=0.1)
    parser.add_argument("--progress_weight", type=float, default=0.1)

    # --- Training ---
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--kl_warmup_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)

    # --- Checkpointing ---
    parser.add_argument("--checkpoint_base", type=str,
                        default="checkpoints/ablation")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save epoch checkpoint every N epochs (for resume)")
    parser.add_argument("--keep_epoch_checkpoints", action="store_true",
                        help="Don't delete epoch checkpoints after variant completes")

    # --- W&B ---
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="tom-compare-v1")
    parser.add_argument("--wandb_entity", type=str, default="nigeldoering-uc-san-diego")

    config = parser.parse_args()

    # Handle --no_wandb flag
    if config.no_wandb:
        config.use_wandb = False

    # --- Seed ---
    set_seed(config.seed)
    device = get_device()

    # --- Determine which variants to train ---
    if config.variants is not None:
        # Map "full" back to None
        selected = []
        for v in config.variants:
            if v == "full":
                selected.append(None)
            elif v in {m for m in ABLATION_VARIANTS if m is not None}:
                selected.append(v)
            else:
                parser.error(f"Unknown variant '{v}'. Choose from: full, "
                             + ", ".join(v for v in ABLATION_VARIANTS if v))
        variants_to_train = selected
    else:
        variants_to_train = list(ABLATION_VARIANTS)

    # --- Handle --restart (delete checkpoint dirs) ---
    if config.restart:
        import shutil
        for v in config.restart:
            mode = None if v == "full" else v
            d = variant_checkpoint_dir(config.checkpoint_base, mode)
            if d.exists():
                shutil.rmtree(d)
                print(f"🗑️  Deleted checkpoint dir for '{v}': {d}")

    # --- Load data once ---
    graph, trajectories, poi_nodes, train_idx, val_idx, num_agents = load_data(
        config.data_dir, config.graph_path, config.split_indices_path,
        config.trajectory_filename,
    )

    # --- Train each variant sequentially ---
    print(f"\n{'=' * 80}")
    print(f"🔬 ABLATION STUDY — {len(variants_to_train)} variants")
    names = [VARIANT_DISPLAY_NAMES[v] for v in variants_to_train]
    print(f"   Variants: {', '.join(names)}")
    print(f"   Epochs per variant: {config.num_epochs}")
    print(f"   Checkpoint base: {config.checkpoint_base}")
    print(f"   W&B: {'enabled' if config.use_wandb and WANDB_AVAILABLE else 'disabled'}")
    print(f"{'=' * 80}")

    overall_t0 = time.time()

    for i, ablation_mode in enumerate(variants_to_train):
        display = VARIANT_DISPLAY_NAMES[ablation_mode]
        print(f"\n{'─' * 80}")
        print(f"  [{i+1}/{len(variants_to_train)}] {display}")
        print(f"{'─' * 80}")

        try:
            train_variant(
                ablation_mode=ablation_mode,
                config=config,
                graph=graph,
                trajectories=trajectories,
                poi_nodes=poi_nodes,
                train_idx=train_idx,
                val_idx=val_idx,
                num_agents=num_agents,
                device=device,
            )
        except Exception as e:
            print(f"\n❌ Variant '{display}' CRASHED: {e}")
            import traceback
            traceback.print_exc()
            print(f"   Continuing to next variant...")
            if WANDB_AVAILABLE and config.use_wandb:
                try:
                    wandb.finish(exit_code=1)
                except Exception:
                    pass
            continue

    total_min = (time.time() - overall_t0) / 60
    print(f"\n{'=' * 80}")
    print(f"🎉 ABLATION STUDY COMPLETE — {total_min:.1f} min total")

    # Summary
    print(f"\n📋 Summary:")
    for v in variants_to_train:
        display = VARIANT_DISPLAY_NAMES[v]
        d = variant_checkpoint_dir(config.checkpoint_base, v)
        done = is_variant_complete(d)
        best = d / "best_model.pt"
        if done:
            info = (d / "DONE").read_text().strip()
            print(f"  ✅ {display:25s} — {info}")
        elif best.exists():
            ckpt = torch.load(best, map_location="cpu")
            acc = ckpt.get("val_metric", 0.0)
            print(f"  ⚠️  {display:25s} — best_model exists (val_acc={acc:.2f}%) but not marked DONE")
        else:
            print(f"  ❌ {display:25s} — no checkpoint found")

    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
