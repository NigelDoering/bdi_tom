"""
Test: Verify that agents reach their goal nodes

This test validates that for every trajectory in the simulation data,
the agent actually reached the final goal node (i.e., the last node in
the path matches the goal_node field).
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_goal_reached(run_id):
    """
    Test that all trajectories end at their goal node.
    
    Args:
        run_id (int): The simulation run ID to test
        
    Returns:
        tuple: (passed, total, failures) where failures is a list of error dicts
    """
    # Load trajectory data
    traj_path = project_root / "data" / "simulation_data" / f"run_{run_id}" / "trajectories" / "all_trajectories.json"
    
    if not traj_path.exists():
        print(f"❌ ERROR: Trajectory file not found: {traj_path}")
        return False, 0, [{"error": "File not found", "path": str(traj_path)}]
    
    with open(traj_path, 'r') as f:
        all_trajectories = json.load(f)
    
    # Track results
    total_trajectories = 0
    passed_trajectories = 0
    failures = []
    
    # Test each trajectory
    for agent_id, agent_trajs in all_trajectories.items():
        for traj_idx, traj in enumerate(agent_trajs):
            total_trajectories += 1
            
            # Extract data
            path = traj.get("path", [])
            goal_node = traj.get("goal_node")
            
            # Handle both annotated (node, goal) tuples and plain node IDs
            if not path:
                failures.append({
                    "agent_id": agent_id,
                    "trajectory_index": traj_idx,
                    "error": "Empty path",
                    "goal_node": goal_node
                })
                continue
            
            # Get last node from path
            last_entry = path[-1]
            if isinstance(last_entry, (list, tuple)):
                # Annotated format: (node, goal)
                last_node = last_entry[0]
            else:
                # Plain node ID format
                last_node = last_entry
            
            # Check if last node matches goal
            if last_node == goal_node:
                passed_trajectories += 1
            else:
                failures.append({
                    "agent_id": agent_id,
                    "trajectory_index": traj_idx,
                    "error": "Last node does not match goal",
                    "last_node": last_node,
                    "goal_node": goal_node,
                    "path_length": len(path)
                })
    
    return passed_trajectories, total_trajectories, failures


def print_results(passed, total, failures):
    """Print test results in a readable format."""
    print(f"\n{'='*60}")
    print(f"Goal Reached Validation Results")
    print(f"{'='*60}")
    print(f"Total trajectories tested: {total}")
    print(f"Passed: {passed} ({100*passed/total:.1f}%)" if total > 0 else "Passed: 0")
    print(f"Failed: {len(failures)} ({100*len(failures)/total:.1f}%)" if total > 0 else "Failed: 0")
    print(f"{'='*60}")
    
    if failures:
        print(f"\n❌ FAILURES DETECTED ({len(failures)} total):")
        print(f"{'='*60}")
        
        # Show first 10 failures in detail
        for i, failure in enumerate(failures[:10], 1):
            print(f"\nFailure {i}:")
            print(f"  Agent ID: {failure.get('agent_id')}")
            print(f"  Trajectory Index: {failure.get('trajectory_index')}")
            print(f"  Error: {failure.get('error')}")
            
            if 'last_node' in failure:
                last_node_display = failure['last_node'][:50] + "..." if len(failure['last_node']) > 50 else failure['last_node']
                goal_node_display = failure['goal_node'][:50] + "..." if len(failure['goal_node']) > 50 else failure['goal_node']
                print(f"  Last Node: {last_node_display}")
                print(f"  Goal Node: {goal_node_display}")
                print(f"  Path Length: {failure.get('path_length')}")
        
        if len(failures) > 10:
            print(f"\n... and {len(failures) - 10} more failures (truncated for readability)")
        
        return False
    else:
        print(f"\n✓ All trajectories successfully reached their goal nodes!")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Test that all trajectories end at their goal node"
    )
    parser.add_argument(
        "--run_id", type=int, required=True,
        help="Simulation run ID to test"
    )
    args = parser.parse_args()
    
    print(f"\nTesting run_{args.run_id}...")
    passed, total, failures = test_goal_reached(args.run_id)
    
    success = print_results(passed, total, failures)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
