"""
Quick test script to verify multi-goal planning logic with retry and home fallback.
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
    """Test the new multi-goal planning with retry logic."""
    
    print("Loading graph...")
    graph_path = os.path.join("data", "processed", "ucsd_walk_full.graphml")
    G = nx.read_graphml(graph_path)
    world_graph = WorldGraph(G)
    
    print("Creating test agent...")
    agent = Agent(agent_id="test_agent", world_graph=world_graph)
    
    print("\n" + "="*60)
    print("Testing plan_path with multi-goal retry logic")
    print("="*60)
    
    # Test at different hours to see variety
    test_hours = [8, 14, 22]  # Morning, afternoon, late night
    
    for hour in test_hours:
        print(f"\n--- Testing at hour {hour}:00 ---")
        
        try:
            result = plan_path(agent, current_hour=hour, temperature=30.0, max_attempts=4)
            
            print(f"✓ Plan successful!")
            print(f"  Start: {result['start_node'][:50]}")
            print(f"  Goal: {result['goal_node'][:50]}")
            print(f"  Attempts: {result['attempts']}")
            print(f"  Returned home: {result['returned_home']}")
            print(f"  Path length: {len(result['path'])} nodes")
            print(f"  Attempted goals: {len(result['attempted_goals'])}")
            
            # Show first few path entries
            print(f"  First 3 path entries:")
            for i, (node, goal) in enumerate(result['path'][:3]):
                print(f"    {i+1}. Node: {node[:40]}... → Goal: {goal[:40]}...")
            
            # Show goal changes
            if result['attempts'] > 1:
                print(f"  Goal progression:")
                for i, goal in enumerate(result['attempted_goals']):
                    node_data = agent.G.nodes[goal]
                    category = node_data.get('Category', 'unknown')
                    hours = node_data.get('opening_hours', {})
                    print(f"    Attempt {i+1}: {category} node {goal[:40]}... (hours: {hours})")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)

if __name__ == "__main__":
    test_multi_goal_planning()
