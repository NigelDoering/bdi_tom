"""
Quick test script to verify plan_path logic.
"""

import os
import sys
import networkx as nx

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from graph_controller.world_graph import WorldGraph
from agent_controller.agent import Agent
from agent_controller.planning_utils import plan_path

def test_multi_goal_planning():
    """Test plan_path at various hours and check goal_open flags."""
    
    print("Loading graph...")
    graph_path = os.path.join("data", "processed", "ucsd_walk_full.graphml")
    G = nx.read_graphml(graph_path)
    world_graph = WorldGraph(G)
    
    print("Creating test agent...")
    agent = Agent(agent_id="test_agent", world_graph=world_graph)
    
    print("\n" + "="*60)
    print("Testing plan_path")
    print("="*60)
    
    # Test at different hours to see variety
    test_hours = [8, 14, 22]  # Morning, afternoon, late night
    
    for hour in test_hours:
        print(f"\n--- Testing at hour {hour}:00 ---")
        
        try:
            result = plan_path(agent, current_hour=hour, temperature=30.0)
            
            print(f"✓ Plan successful!")
            print(f"  Start: {result['start_node'][:50]}")
            print(f"  Goal: {result['goal_node'][:50]}")
            print(f"  Category: {result.get('goal_category', 'unknown')}")
            print(f"  Goal open: {result.get('goal_open', 'N/A')}")
            print(f"  Path length: {len(result['path'])} nodes")
            
            # Show first few path entries
            print(f"  First 3 path entries:")
            for i, (node, goal) in enumerate(result['path'][:3]):
                print(f"    {i+1}. Node: {node[:40]}... → Goal: {goal[:40]}...")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)

if __name__ == "__main__":
    test_multi_goal_planning()
