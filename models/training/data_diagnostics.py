"""
Data distribution diagnostic tools for debugging train/val distribution mismatch.
"""

import numpy as np
from collections import Counter
from typing import List, Dict


def analyze_data_distribution(train_trajs: List[Dict], val_trajs: List[Dict], test_trajs: List[Dict]):
    """
    Rigorous analysis of distribution differences between splits.
    
    This helps identify:
    - Temporal biases (certain hours only in train/val)
    - Agent distribution issues
    - Goal distribution mismatches
    - Trajectory length differences
    - Spatial biases
    """
    
    print("\n" + "="*100)
    print(" "*35 + "DATA DISTRIBUTION ANALYSIS")
    print("="*100)
    
    # ========================================
    # 1. HOUR DISTRIBUTION
    # ========================================
    print("\n📅 TEMPORAL DISTRIBUTION (Hour of Day):")
    print("-" * 100)
    
    train_hours = [t['hour'] for t in train_trajs]
    val_hours = [t['hour'] for t in val_trajs]
    test_hours = [t['hour'] for t in test_trajs]
    
    print(f"   Train: min={min(train_hours):2d}, max={max(train_hours):2d}, "
          f"mean={np.mean(train_hours):5.2f}, std={np.std(train_hours):5.2f}")
    print(f"   Val:   min={min(val_hours):2d}, max={max(val_hours):2d}, "
          f"mean={np.mean(val_hours):5.2f}, std={np.std(val_hours):5.2f}")
    print(f"   Test:  min={min(test_hours):2d}, max={max(test_hours):2d}, "
          f"mean={np.mean(test_hours):5.2f}, std={np.std(test_hours):5.2f}")
    
    # Distribution comparison
    train_hour_dist = np.bincount(train_hours, minlength=24) / len(train_hours)
    val_hour_dist = np.bincount(val_hours, minlength=24) / len(val_hours)
    
    # KL divergence
    kl_div = np.sum(np.where(train_hour_dist > 0, 
                             train_hour_dist * np.log((train_hour_dist + 1e-10) / (val_hour_dist + 1e-10)), 
                             0))
    
    status = "🚨 SEVERE!" if kl_div > 0.5 else ("⚠️ HIGH!" if kl_div > 0.1 else "✅ OK")
    print(f"\n   KL Divergence (train||val): {kl_div:.4f} {status}")
    
    if kl_div > 0.1:
        print(f"   ⚠️ Warning: Hour distributions differ significantly!")
        print(f"   Top 5 hours in train: {np.argsort(train_hour_dist)[-5:][::-1].tolist()}")
        print(f"   Top 5 hours in val:   {np.argsort(val_hour_dist)[-5:][::-1].tolist()}")
    
    # ========================================
    # 2. AGENT DISTRIBUTION
    # ========================================
    print("\n👤 AGENT DISTRIBUTION:")
    print("-" * 100)
    
    train_agents = [t['agent_id'] for t in train_trajs]
    val_agents = [t['agent_id'] for t in val_trajs]
    test_agents = [t['agent_id'] for t in test_trajs]
    
    train_agent_counts = Counter(train_agents)
    val_agent_counts = Counter(val_agents)
    test_agent_counts = Counter(test_agents)
    
    print(f"   Train: {len(train_agent_counts):3d} unique agents, "
          f"avg {np.mean(list(train_agent_counts.values())):6.1f} trajs/agent "
          f"(min={min(train_agent_counts.values())}, max={max(train_agent_counts.values())})")
    print(f"   Val:   {len(val_agent_counts):3d} unique agents, "
          f"avg {np.mean(list(val_agent_counts.values())):6.1f} trajs/agent "
          f"(min={min(val_agent_counts.values())}, max={max(val_agent_counts.values())})")
    print(f"   Test:  {len(test_agent_counts):3d} unique agents, "
          f"avg {np.mean(list(test_agent_counts.values())):6.1f} trajs/agent "
          f"(min={min(test_agent_counts.values())}, max={max(test_agent_counts.values())})")
    
    # Check for agents only in one split
    train_only = set(train_agents) - set(val_agents)
    val_only = set(val_agents) - set(train_agents)
    
    if train_only:
        print(f"   ⚠️ {len(train_only)} agents appear ONLY in train")
    if val_only:
        print(f"   ⚠️ {len(val_only)} agents appear ONLY in validation")
    
    # Check distribution of trajectories per agent
    train_agent_freqs = np.array(list(train_agent_counts.values())) / len(train_trajs)
    val_agent_freqs = np.array(list(val_agent_counts.values())) / len(val_trajs)
    
    print(f"\n   Agent frequency entropy:")
    train_entropy = -np.sum(train_agent_freqs * np.log(train_agent_freqs + 1e-10))
    val_entropy = -np.sum(val_agent_freqs * np.log(val_agent_freqs + 1e-10))
    print(f"   Train: {train_entropy:.4f}")
    print(f"   Val:   {val_entropy:.4f}")
    print(f"   Difference: {abs(train_entropy - val_entropy):.4f} "
          f"{'⚠️ HIGH' if abs(train_entropy - val_entropy) > 0.5 else '✅ OK'}")
    
    # ========================================
    # 3. GOAL DISTRIBUTION
    # ========================================
    print("\n🎯 GOAL DISTRIBUTION:")
    print("-" * 100)
    
    train_goals = [t['goal_node'] for t in train_trajs]
    val_goals = [t['goal_node'] for t in val_trajs]
    test_goals = [t['goal_node'] for t in test_trajs]
    
    train_goal_counts = Counter(train_goals)
    val_goal_counts = Counter(val_goals)
    test_goal_counts = Counter(test_goals)
    
    print(f"   Train: {len(train_goal_counts):3d} unique goals")
    print(f"   Val:   {len(val_goal_counts):3d} unique goals")
    print(f"   Test:  {len(test_goal_counts):3d} unique goals")
    
    # Check for goals only in one split
    train_only_goals = set(train_goals) - set(val_goals)
    val_only_goals = set(val_goals) - set(train_goals)
    
    if train_only_goals:
        print(f"\n   ⚠️ {len(train_only_goals)} goals appear ONLY in train")
        print(f"      Examples: {list(train_only_goals)[:5]}")
        print(f"      Total trajectories: {sum(train_goal_counts[g] for g in train_only_goals)}")
    
    if val_only_goals:
        print(f"\n   🚨 CRITICAL: {len(val_only_goals)} goals appear ONLY in validation!")
        print(f"      Examples: {list(val_only_goals)[:5]}")
        print(f"      Total trajectories: {sum(val_goal_counts[g] for g in val_only_goals)}")
        print(f"      ⚠️ Model has NEVER seen these goals during training!")
    
    # Compare frequencies for shared goals
    shared_goals = set(train_goals) & set(val_goals)
    print(f"\n   Shared goals: {len(shared_goals)}")
    
    if shared_goals:
        freq_diffs = []
        max_diff_goal = None
        max_diff = 0
        
        for goal in shared_goals:
            train_freq = train_goal_counts[goal] / len(train_goals)
            val_freq = val_goal_counts[goal] / len(val_goals)
            diff = abs(train_freq - val_freq)
            freq_diffs.append(diff)
            
            if diff > max_diff:
                max_diff = diff
                max_diff_goal = goal
        
        print(f"   Mean frequency difference: {np.mean(freq_diffs):.4f}")
        print(f"   Max frequency difference: {np.max(freq_diffs):.4f} (goal: {max_diff_goal})")
        
        if np.mean(freq_diffs) > 0.01:
            print(f"   ⚠️ Warning: Goal frequencies differ significantly between train/val")
    
    # Goal frequency entropy
    train_goal_freqs = np.array(list(train_goal_counts.values())) / len(train_goals)
    val_goal_freqs = np.array(list(val_goal_counts.values())) / len(val_goals)
    
    train_goal_entropy = -np.sum(train_goal_freqs * np.log(train_goal_freqs + 1e-10))
    val_goal_entropy = -np.sum(val_goal_freqs * np.log(val_goal_freqs + 1e-10))
    
    print(f"\n   Goal distribution entropy:")
    print(f"   Train: {train_goal_entropy:.4f}")
    print(f"   Val:   {val_goal_entropy:.4f}")
    print(f"   Difference: {abs(train_goal_entropy - val_goal_entropy):.4f}")
    
    # ========================================
    # 4. TRAJECTORY LENGTH DISTRIBUTION
    # ========================================
    print("\n📏 TRAJECTORY LENGTH DISTRIBUTION:")
    print("-" * 100)
    
    train_lens = [len(t['path']) for t in train_trajs]
    val_lens = [len(t['path']) for t in val_trajs]
    test_lens = [len(t['path']) for t in test_trajs]
    
    print(f"   Train: mean={np.mean(train_lens):6.1f}, median={np.median(train_lens):6.1f}, "
          f"std={np.std(train_lens):6.1f}, min={min(train_lens):3d}, max={max(train_lens):3d}")
    print(f"   Val:   mean={np.mean(val_lens):6.1f}, median={np.median(val_lens):6.1f}, "
          f"std={np.std(val_lens):6.1f}, min={min(val_lens):3d}, max={max(val_lens):3d}")
    print(f"   Test:  mean={np.mean(test_lens):6.1f}, median={np.median(test_lens):6.1f}, "
          f"std={np.std(test_lens):6.1f}, min={min(test_lens):3d}, max={max(test_lens):3d}")
    
    # Statistical test
    from scipy import stats
    ks_stat, ks_p = stats.ks_2samp(train_lens, val_lens)
    print(f"\n   Kolmogorov-Smirnov test (train vs val):")
    print(f"   Statistic: {ks_stat:.4f}, p-value: {ks_p:.4f}")
    if ks_p < 0.05: # type: ignore
        print(f"   🚨 CRITICAL: Length distributions are significantly different!")
    else:
        print(f"   ✅ Length distributions are similar")
    
    # ========================================
    # 5. START NODE DISTRIBUTION
    # ========================================
    print("\n🚀 START NODE DISTRIBUTION:")
    print("-" * 100)
    
    def get_start_node(traj):
        path = traj['path']
        if isinstance(path[0], (list, tuple)):
            return path[0][0]
        return path[0]
    
    train_starts = [get_start_node(t) for t in train_trajs]
    val_starts = [get_start_node(t) for t in val_trajs]
    
    train_start_counts = Counter(train_starts)
    val_start_counts = Counter(val_starts)
    
    print(f"   Train: {len(train_start_counts)} unique start nodes")
    print(f"   Val:   {len(val_start_counts)} unique start nodes")
    
    shared_starts = set(train_starts) & set(val_starts)
    print(f"   Shared: {len(shared_starts)} start nodes")
    
    train_only_starts = set(train_starts) - set(val_starts)
    val_only_starts = set(val_starts) - set(train_starts)
    
    if train_only_starts:
        print(f"   ⚠️ {len(train_only_starts)} start nodes ONLY in train")
    if val_only_starts:
        print(f"   ⚠️ {len(val_only_starts)} start nodes ONLY in validation")
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "="*100)
    print(" "*40 + "SUMMARY")
    print("="*100)
    
    issues = []
    
    if kl_div > 0.1:
        issues.append("🚨 Hour distribution mismatch")
    if train_only or val_only:
        issues.append("⚠️ Agent distribution imbalance")
    if val_only_goals:
        issues.append(f"🚨 CRITICAL: {len(val_only_goals)} goals only in validation")
    if train_only_goals and len(train_only_goals) > 5:
        issues.append(f"⚠️ {len(train_only_goals)} goals only in train")
    if ks_p < 0.05: # type: ignore
        issues.append("🚨 Trajectory length distribution mismatch")
    
    if issues:
        print("\n⚠️ POTENTIAL DISTRIBUTION ISSUES FOUND:")
        for issue in issues:
            print(f"   {issue}")
        print("\nThese distribution mismatches could explain the immediate train/val divergence!")
    else:
        print("\n✅ No major distribution issues found.")
        print("The train/val/test splits appear to come from the same distribution.")
        print("Overfitting is likely due to model capacity or regularization, not data issues.")
    
    print("\n" + "="*100 + "\n")
