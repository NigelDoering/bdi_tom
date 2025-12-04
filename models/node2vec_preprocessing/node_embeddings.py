"""
Node2Vec-based node embeddings for graph representation.

This module provides Node2Vec training and embedding management for the BDI-ToM models.
All nodes in the graph are embedded into a shared vector space that is used by both
trajectory and map encoders.
"""

import os
import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle

from node2vec import Node2Vec
 



class Node2VecEmbeddings:
    """
    Node2Vec embeddings manager for graph nodes.
    
    This class handles:
    - Training Node2Vec on a graph
    - Saving/loading embeddings to/from disk
    - Providing embeddings for nodes
    - Converting node sequences to embedding matrices
    
    Attributes:
        embedding_dim: Dimension of node embeddings
        embeddings: Dict mapping node IDs to embedding vectors
        node_to_idx: Dict mapping node IDs to integer indices
        idx_to_node: Dict mapping integer indices to node IDs
        embedding_matrix: Tensor of shape (num_nodes, embedding_dim)
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        window_size: int = 10,
        num_workers: int = 4,
        seed: int = 42
    ):
        """
        Initialize Node2Vec embeddings manager.
        
        Args:
            embedding_dim: Dimension of node embeddings
            walk_length: Length of random walks
            num_walks: Number of walks per node
            p: Return parameter (controls likelihood of immediately revisiting a node)
            q: In-out parameter (controls search to be DFS-like or BFS-like)
            window_size: Context window size for skip-gram
            num_workers: Number of parallel workers
            seed: Random seed for reproducibility
        """
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.window_size = window_size
        self.num_workers = num_workers
        self.seed = seed
        
        # Will be populated after training or loading
        self.embeddings: Optional[Dict[str, np.ndarray]] = None
        self.node_to_idx: Optional[Dict[str, int]] = None
        self.idx_to_node: Optional[Dict[int, str]] = None
        self.embedding_matrix: Optional[torch.Tensor] = None
        self._num_nodes: Optional[int] = None
    
    def train(self, graph: nx.Graph, verbose: bool = True) -> None:
        """
        Train Node2Vec on a graph.
        
        Args:
            graph: NetworkX graph to train on
            verbose: Whether to print progress
        """
        
        if verbose:
            print(f"\n🎯 Training Node2Vec embeddings...")
            print(f"   Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            print(f"   Embedding dim: {self.embedding_dim}")
            print(f"   Walk length: {self.walk_length}, Num walks: {self.num_walks}")
            print(f"   p={self.p}, q={self.q}, window={self.window_size}")
        
        # Initialize Node2Vec
        node2vec = Node2Vec(
            graph,
            dimensions=self.embedding_dim,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            p=self.p,
            q=self.q,
            workers=self.num_workers,
            seed=self.seed,
            quiet=not verbose
        )
        
        # Train embeddings
        if verbose:
            print("   Training Word2Vec model...")
        
        model = node2vec.fit(
            window=self.window_size,
            min_count=1,
            batch_words=4,
            seed=self.seed
        )
        
        # Extract embeddings
        self.embeddings = {}
        nodes = list(graph.nodes())
        
        for node in nodes:
            node_str = str(node)
            self.embeddings[node] = model.wv[node_str]

        
        # Create node mappings
        self.node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        self._num_nodes = len(nodes)
        
        # Create embedding matrix
        self._create_embedding_matrix()
        
        if verbose:
            print(f"   ✅ Successfully trained embeddings for {len(self.embeddings)} nodes")
    
    def _create_embedding_matrix(self) -> None:
        """Create embedding matrix from embeddings dict."""
        if self.embeddings is None or self.node_to_idx is None:
            raise RuntimeError("Embeddings not trained or loaded yet")
        
        # Create matrix
        num_nodes = len(self.node_to_idx)
        embedding_matrix = np.zeros((num_nodes, self.embedding_dim), dtype=np.float32)
        
        for node, idx in self.node_to_idx.items():
            embedding_matrix[idx] = self.embeddings[node]
        
        self.embedding_matrix = torch.from_numpy(embedding_matrix)
    
    def save(self, save_path: str) -> None:
        """
        Save embeddings to disk.
        
        Args:
            save_path: Path to save embeddings (will save as .pkl)
        """
        if self.embeddings is None:
            raise RuntimeError("No embeddings to save. Train or load embeddings first.")
        
        save_path = Path(save_path) # type: ignore
        save_path.parent.mkdir(parents=True, exist_ok=True) # type: ignore
        
        # Save all necessary data
        data = {
            'embeddings': self.embeddings,
            'node_to_idx': self.node_to_idx,
            'idx_to_node': self.idx_to_node,
            'embedding_dim': self.embedding_dim,
            'config': {
                'walk_length': self.walk_length,
                'num_walks': self.num_walks,
                'p': self.p,
                'q': self.q,
                'window_size': self.window_size,
                'seed': self.seed
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Saved embeddings to {save_path}")
    
    def load(self, load_path: str, verbose: bool = True) -> None:
        """
        Load embeddings from disk.
        
        Args:
            load_path: Path to load embeddings from
            verbose: Whether to print progress
        """
        load_path = Path(load_path) # type: ignore
        
        if not load_path.exists(): # type: ignore
            raise FileNotFoundError(f"Embeddings file not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.node_to_idx = data['node_to_idx']
        self.idx_to_node = data['idx_to_node']
        self.embedding_dim = data['embedding_dim']
        self._num_nodes = len(self.node_to_idx) # type: ignore
        
        # Restore config
        config = data.get('config', {})
        self.walk_length = config.get('walk_length', self.walk_length)
        self.num_walks = config.get('num_walks', self.num_walks)
        self.p = config.get('p', self.p)
        self.q = config.get('q', self.q)
        self.window_size = config.get('window_size', self.window_size)
        self.seed = config.get('seed', self.seed)
        
        # Create embedding matrix
        self._create_embedding_matrix()
        
        if verbose:
            print(f"✅ Loaded embeddings from {load_path}")
            print(f"   {len(self.embeddings)} nodes, {self.embedding_dim} dimensions") # type: ignore
    
    def get_embedding(self, node_id) -> torch.Tensor:
        """
        Get embedding for a single node.
        
        Args:
            node_id: Node identifier
        
        Returns:
            Tensor of shape (embedding_dim,)
        """
        if self.embeddings is None:
            raise RuntimeError("Embeddings not trained or loaded yet")
        
        if node_id not in self.embeddings:
            raise KeyError(f"Node {node_id} not found in embeddings")
        
        return torch.from_numpy(self.embeddings[node_id])
    
    def get_embeddings_batch(self, node_ids: List) -> torch.Tensor:
        """
        Get embeddings for a batch of nodes.
        
        Args:
            node_ids: List of node identifiers
        
        Returns:
            Tensor of shape (len(node_ids), embedding_dim)
        """
        if self.embeddings is None:
            raise RuntimeError("Embeddings not trained or loaded yet")
        
        embeddings = []
        for node_id in node_ids:
            if node_id not in self.embeddings:
                raise KeyError(f"Node {node_id} not found in embeddings")
            embeddings.append(self.embeddings[node_id])
        
        return torch.from_numpy(np.stack(embeddings))
    
    def trajectory_to_embeddings(self, trajectory: List) -> torch.Tensor:
        """
        Convert a trajectory (sequence of node IDs) to embedding matrix.
        
        Args:
            trajectory: List of node identifiers
        
        Returns:
            Tensor of shape (len(trajectory), embedding_dim)
        """
        return self.get_embeddings_batch(trajectory)
    
    def trajectories_to_embeddings(
        self,
        trajectories: List[List],
        pad_value: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert multiple trajectories to padded embedding matrices.
        
        Args:
            trajectories: List of trajectories (each a list of node IDs)
            pad_value: Value to use for padding
        
        Returns:
            Tuple of:
            - embeddings: Tensor of shape (batch_size, max_len, embedding_dim)
            - lengths: Tensor of shape (batch_size,) with actual trajectory lengths
        """
        if self.embeddings is None:
            raise RuntimeError("Embeddings not trained or loaded yet")
        
        batch_size = len(trajectories)
        max_len = max(len(traj) for traj in trajectories)
        
        # Initialize with pad value
        embeddings = torch.full(
            (batch_size, max_len, self.embedding_dim),
            pad_value,
            dtype=torch.float32
        )
        
        lengths = torch.zeros(batch_size, dtype=torch.long)
        
        for i, traj in enumerate(trajectories):
            traj_len = len(traj)
            lengths[i] = traj_len
            
            # Get embeddings for this trajectory
            traj_embeddings = self.trajectory_to_embeddings(traj)
            embeddings[i, :traj_len] = traj_embeddings
        
        return embeddings, lengths
    
    def get_node_index(self, node_id) -> int:
        """
        Get integer index for a node.
        
        Args:
            node_id: Node identifier
        
        Returns:
            Integer index
        """
        if self.node_to_idx is None:
            raise RuntimeError("Embeddings not trained or loaded yet")
        
        if node_id not in self.node_to_idx:
            raise KeyError(f"Node {node_id} not found in node_to_idx mapping")
        
        return self.node_to_idx[node_id]
    
    def get_node_indices(self, node_ids: List) -> torch.Tensor:
        """
        Get integer indices for multiple nodes.
        
        Args:
            node_ids: List of node identifiers
        
        Returns:
            Tensor of shape (len(node_ids),) with integer indices
        """
        indices = [self.get_node_index(node_id) for node_id in node_ids]
        return torch.tensor(indices, dtype=torch.long)
    
    @property
    def num_nodes(self) -> int:
        """Get number of nodes."""
        if self._num_nodes is None:
            raise RuntimeError("Embeddings not trained or loaded yet")
        return self._num_nodes
    
    def __len__(self) -> int:
        """Get number of nodes."""
        return self.num_nodes


def get_or_create_embeddings(
    graph: nx.Graph,
    cache_path: str,
    embedding_dim: int = 128,
    force_retrain: bool = False,
    **node2vec_kwargs
) -> Node2VecEmbeddings:
    """
    Get embeddings from cache or create new ones.
    
    This is a convenience function that checks if embeddings exist on disk,
    loads them if they do, or trains new ones if they don't.
    
    Args:
        graph: NetworkX graph
        cache_path: Path to cache embeddings
        embedding_dim: Dimension of embeddings
        force_retrain: If True, retrain even if cache exists
        **node2vec_kwargs: Additional arguments for Node2VecEmbeddings
    
    Returns:
        Node2VecEmbeddings instance with trained/loaded embeddings
    """
    cache_path = Path(cache_path) # type: ignore
    
    embeddings = Node2VecEmbeddings(embedding_dim=embedding_dim, **node2vec_kwargs)
    
    if cache_path.exists() and not force_retrain: # type: ignore
        print(f"📂 Loading cached embeddings from {cache_path}")
        embeddings.load(str(cache_path))
    else:
        if force_retrain:
            print(f"🔄 Force retraining embeddings...")
        else:
            print(f"💾 No cached embeddings found, training new ones...")
        
        embeddings.train(graph)
        embeddings.save(str(cache_path))
    
    return embeddings
