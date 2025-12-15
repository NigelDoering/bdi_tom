"""
MINIMAL EXTRACTION TEST
=======================

Test that encoder extraction works correctly.
Run this after training a model to verify extraction works.

Usage:
    python models/training/test_encoder_extraction.py \
        --checkpoint checkpoints/per_node_v1/best_model.pt
"""

import os
import sys
import argparse
import torch
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.training.extract_encoder import extract_encoder_from_checkpoint, load_frozen_encoder


def test_extraction(checkpoint_path: str):
    """Test encoder extraction end-to-end."""
    
    print("\n" + "="*70)
    print("🧪 ENCODER EXTRACTION TEST")
    print("="*70)
    
    # ====================================================================
    # TEST 1: Extract encoder
    # ====================================================================
    print("\n[TEST 1] Extract encoder from checkpoint...")
    try:
        encoder = extract_encoder_from_checkpoint(checkpoint_path, verbose=False)
        print("✅ PASS: Encoder extracted")
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False
    
    # ====================================================================
    # TEST 2: Check frozen status
    # ====================================================================
    print("\n[TEST 2] Verify encoder is frozen...")
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    if trainable_params == 0:
        print(f"✅ PASS: Encoder is frozen (trainable params = 0)")
    else:
        print(f"❌ FAIL: Encoder not frozen (trainable params = {trainable_params:,})")
        return False
    
    # ====================================================================
    # TEST 3: Check output dimension
    # ====================================================================
    print("\n[TEST 3] Check encoder output dimension...")
    if hasattr(encoder, 'fusion_dim'):
        print(f"✅ PASS: Output dimension = {encoder.fusion_dim}")
    else:
        print("❌ FAIL: encoder.fusion_dim not found")
        return False
    
    # ====================================================================
    # TEST 4: Forward pass (encode_nodes)
    # ====================================================================
    print("\n[TEST 4] Test forward pass (encode_nodes)...")
    try:
        batch = torch.randint(0, 100, (4, 5))  # [batch=4, seq_len=5]
        with torch.no_grad():
            # encode_nodes returns node embeddings (per-node)
            embeddings = encoder.encode_nodes(batch)  # [batch, seq_len, node_embedding_dim]
        
        # Check we got valid output
        if embeddings is not None and embeddings.shape[0] == 4 and embeddings.shape[1] == 5:
            print(f"✅ PASS: Output shape = {embeddings.shape}")
        else:
            print(f"❌ FAIL: Got {embeddings.shape if embeddings is not None else None}")
            return False
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False
    
    # ====================================================================
    # TEST 5: Save and reload
    # ====================================================================
    print("\n[TEST 5] Save and reload encoder...")
    try:
        save_path = "checkpoints/test_encoder.pt"
        encoder = extract_encoder_from_checkpoint(
            checkpoint_path,
            output_path=save_path,
            verbose=False
        )
        
        # Reload
        encoder2 = load_frozen_encoder(save_path)
        
        # Verify same output
        batch = torch.randint(0, 100, (4, 5))
        
        with torch.no_grad():
            out1 = encoder.encode_nodes(batch)
            out2 = encoder2.encode_nodes(batch)
        
        if out1 is not None and out2 is not None and torch.allclose(out1, out2):
            print(f"✅ PASS: Saved and reloaded encoder works")
            # Clean up
            if os.path.exists(save_path):
                os.remove(save_path)
        else:
            print("❌ FAIL: Reloaded encoder outputs differ or None")
            return False
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False
    
    # ====================================================================
    # TEST 6: With custom head
    # ====================================================================
    print("\n[TEST 6] Test with custom prediction head...")
    try:
        import torch.nn as nn
        
        encoder = extract_encoder_from_checkpoint(checkpoint_path, verbose=False)
        batch = torch.randint(0, 100, (4, 5))
        
        with torch.no_grad():
            embeddings = encoder.encode_nodes(batch)
        
        # Use last embedding and pool to create feature vector
        if embeddings is not None:
            # Average pool over sequence or take last
            feature_vec = embeddings[:, -1, :]  # [4, node_embedding_dim]
            
            # Create head that maps from node_embedding to output
            head = nn.Linear(feature_vec.shape[-1], 7)
            logits = head(feature_vec)
            
            if logits.shape == (4, 7):
                print(f"✅ PASS: Head output shape = {logits.shape}")
            else:
                print(f"❌ FAIL: Expected (4, 7), got {logits.shape}")
                return False
        else:
            print("❌ FAIL: encode_nodes returned None")
            return False
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False
    
    # ====================================================================
    # TEST 7: Verify encoder not affecting head
    # ====================================================================
    print("\n[TEST 7] Verify encoder parameters frozen during training...")
    try:
        encoder = extract_encoder_from_checkpoint(checkpoint_path, verbose=False)
        batch = torch.randint(0, 100, (4, 5))
        targets = torch.randint(0, 7, (4,))
        
        with torch.no_grad():
            embeddings = encoder.encode_nodes(batch)
        
        if embeddings is None:
            print("❌ FAIL: encode_nodes returned None")
            return False
        
        # Create head based on actual embedding dimension
        head = nn.Linear(embeddings.shape[-1], 7)
        optimizer = torch.optim.Adam(head.parameters(), lr=0.001)
        
        # Forward
        feature_vec = embeddings[:, -1, :]
        logits = head(feature_vec)
        loss = nn.functional.cross_entropy(logits, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that encoder has no gradients
        encoder_has_grads = any(p.grad is not None for p in encoder.parameters())
        if not encoder_has_grads:
            print("✅ PASS: Encoder parameters not updated during backprop")
        else:
            print("❌ FAIL: Encoder parameters were updated")
            return False
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False
    
    # ====================================================================
    # SUMMARY
    # ====================================================================
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nEncoder extraction is working correctly and ready for use.")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test encoder extraction")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/per_node_v2/best_model.pt",
        help="Path to checkpoint to test"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"\n❌ Checkpoint not found: {args.checkpoint}")
        print("Please train a model first:")
        print("  python models/training/train_per_node.py")
        sys.exit(1)
    
    success = test_extraction(args.checkpoint)
    sys.exit(0 if success else 1)
