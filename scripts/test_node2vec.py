"""
Test script for Node2Vec embeddings.

This script verifies that Node2Vec embeddings can be trained, saved, and loaded correctly.
"""

import os
import sys
import networkx as nx

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.data.node_embeddings import Node2VecEmbeddings, get_or_create_embeddings


def main():
    """Test Node2Vec embeddings."""
    print("\n" + "=" * 80)
    print("NODE2VEC EMBEDDINGS TEST")
    print("=" * 80 + "\n")
    
    # Load the UCSD graph
    print("📂 Loading graph...")
    graph_path = "data/processed/ucsd_walk_full.graphml"
    
    if not os.path.exists(graph_path):
        print(f"❌ Graph not found at {graph_path}")
        return
    
    graph = nx.read_graphml(graph_path)
    print(f"✅ Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Test 1: Train embeddings
    print("\n" + "-" * 80)
    print("TEST 1: Training Node2Vec embeddings")
    print("-" * 80)
    
    embeddings = Node2VecEmbeddings(
        embedding_dim=128,
        walk_length=30,
        num_walks=10,
        p=1.0,
        q=1.0,
        window_size=10,
        num_workers=4,
        seed=42
    )
    
    embeddings.train(graph, verbose=True)
    
    # Test 2: Get embedding for single node
    print("\n" + "-" * 80)
    print("TEST 2: Getting embedding for single node")
    print("-" * 80)
    
    first_node = list(graph.nodes())[0]
    print(f"First node ID: {first_node}")
    
    node_emb = embeddings.get_embedding(first_node)
    print(f"Embedding shape: {node_emb.shape}")
    print(f"Embedding sample (first 10 dims): {node_emb[:10]}")
    
    # Test 3: Get embeddings for multiple nodes
    print("\n" + "-" * 80)
    print("TEST 3: Getting embeddings for batch of nodes")
    print("-" * 80)
    
    sample_nodes = list(graph.nodes())[:5]
    print(f"Sample nodes: {sample_nodes}")
    
    batch_embs = embeddings.get_embeddings_batch(sample_nodes)
    print(f"Batch embeddings shape: {batch_embs.shape}")
    
    # Test 4: Convert trajectory to embeddings
    print("\n" + "-" * 80)
    print("TEST 4: Converting trajectory to embeddings")
    print("-" * 80)
    
    sample_trajectory = list(graph.nodes())[:10]
    print(f"Trajectory length: {len(sample_trajectory)}")
    
    traj_embs = embeddings.trajectory_to_embeddings(sample_trajectory)
    print(f"Trajectory embeddings shape: {traj_embs.shape}")
    
    # Test 5: Convert multiple trajectories with padding
    print("\n" + "-" * 80)
    print("TEST 5: Converting multiple trajectories with padding")
    print("-" * 80)
    
    trajectories = [
        list(graph.nodes())[:5],
        list(graph.nodes())[:10],
        list(graph.nodes())[:3]
    ]
    print(f"Trajectory lengths: {[len(t) for t in trajectories]}")
    
    traj_matrix, lengths = embeddings.trajectories_to_embeddings(trajectories)
    print(f"Padded trajectories shape: {traj_matrix.shape}")
    print(f"Actual lengths: {lengths}")
    
    # Test 6: Save embeddings
    print("\n" + "-" * 80)
    print("TEST 6: Saving embeddings")
    print("-" * 80)
    
    save_path = "data/processed/node2vec_embeddings.pkl"
    embeddings.save(save_path)
    
    # Test 7: Load embeddings
    print("\n" + "-" * 80)
    print("TEST 7: Loading embeddings")
    print("-" * 80)
    
    loaded_embeddings = Node2VecEmbeddings()
    loaded_embeddings.load(save_path, verbose=True)
    
    # Verify loaded embeddings match
    loaded_emb = loaded_embeddings.get_embedding(first_node)
    print(f"Original embedding: {node_emb[:5]}")
    print(f"Loaded embedding: {loaded_emb[:5]}")
    print(f"Match: {(node_emb - loaded_emb).abs().sum() < 1e-6}")
    
    # Test 8: Use convenience function
    print("\n" + "-" * 80)
    print("TEST 8: Using get_or_create_embeddings()")
    print("-" * 80)
    
    cache_path = "data/processed/node2vec_embeddings_cached.pkl"
    
    # First call: should load from existing cache
    embs1 = get_or_create_embeddings(
        graph,
        cache_path,
        embedding_dim=128,
        force_retrain=False
    )
    
    # Second call: should force retrain
    print("\n🔄 Testing force retrain...")
    embs2 = get_or_create_embeddings(
        graph,
        cache_path,
        embedding_dim=128,
        force_retrain=True
    )
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
