"""
Test script to verify accuracy metric computation is correct.

This script tests the compute_accuracy function with known examples
to ensure top-1 and top-5 accuracy are calculated correctly.
"""
import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.training.utils import compute_accuracy


def test_top1_accuracy():
    """Test top-1 accuracy (exact match)."""
    print("\n" + "="*60)
    print("TEST 1: Top-1 Accuracy")
    print("="*60)
    
    # Create a batch of 5 samples with 10 possible classes
    # predictions[i, j] = logit for class j in sample i
    predictions = torch.tensor([
        # Sample 0: highest is class 2 (correct!)
        [0.1, 0.2, 0.9, 0.3, 0.15, 0.1, 0.05, 0.4, 0.25, 0.1],
        
        # Sample 1: highest is class 5 (correct!)
        [0.5, 0.2, 0.1, 0.3, 0.15, 0.8, 0.05, 0.4, 0.25, 0.1],
        
        # Sample 2: highest is class 9 (correct!)
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9],
        
        # Sample 3: highest is class 1, but true is 0 (WRONG!)
        [0.05, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        
        # Sample 4: highest is class 3, but true is 7 (WRONG!)
        [0.1, 0.2, 0.3, 0.9, 0.4, 0.5, 0.6, 0.7, 0.8, 0.05],
    ])
    
    targets = torch.tensor([2, 5, 9, 0, 7])
    
    # Manually verify top-1 predictions
    top1_preds = predictions.argmax(dim=-1)
    print(f"Top-1 predictions: {top1_preds.tolist()}")
    print(f"True targets:      {targets.tolist()}")
    print(f"Matches:           {(top1_preds == targets).tolist()}")
    
    # Expected: 3 correct out of 5 = 60%
    expected = 60.0
    
    # Compute using our function
    accuracy = compute_accuracy(predictions, targets, k=1)
    
    print(f"\n✓ Expected top-1 accuracy: {expected}%")
    print(f"✓ Computed top-1 accuracy: {accuracy}%")
    
    assert abs(accuracy - expected) < 0.01, f"FAILED: Expected {expected}%, got {accuracy}%"
    print("✅ TEST PASSED!\n")


def test_top5_accuracy():
    """Test top-5 accuracy (true class in top 5)."""
    print("="*60)
    print("TEST 2: Top-5 Accuracy")
    print("="*60)
    
    # Create a batch where some targets are in top-5, some aren't
    predictions = torch.tensor([
        # Sample 0: True class 2 is rank #1 (in top-5 ✓)
        [0.1, 0.2, 0.9, 0.3, 0.15, 0.1, 0.05, 0.4, 0.25, 0.1],
        
        # Sample 1: True class 5 is rank #1 (in top-5 ✓)
        [0.5, 0.2, 0.1, 0.3, 0.15, 0.8, 0.05, 0.4, 0.25, 0.1],
        
        # Sample 2: True class 9 is rank #1 (in top-5 ✓)
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9],
        
        # Sample 3: True class 0 is rank #10 (NOT in top-5 ✗)
        # Top-5: [1, 2, 3, 4, 5] - class 0 has lowest score
        [0.05, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        
        # Sample 4: True class 7 is rank #3 (in top-5 ✓)
        # Top-5: [8, 6, 7, 5, 4]
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.05],
    ])
    
    targets = torch.tensor([2, 5, 9, 0, 7])
    
    # Manually verify top-5 predictions
    _, top5_indices = predictions.topk(5, dim=-1)
    print(f"Top-5 predictions (indices):")
    for i, (pred5, target) in enumerate(zip(top5_indices, targets)):
        in_top5 = target.item() in pred5.tolist()
        status = "✓" if in_top5 else "✗"
        print(f"  Sample {i}: {pred5.tolist()} | True: {target.item()} | {status}")
    
    # Expected: 4 correct out of 5 = 80%
    expected = 80.0
    
    # Compute using our function
    accuracy = compute_accuracy(predictions, targets, k=5)
    
    print(f"\n✓ Expected top-5 accuracy: {expected}%")
    print(f"✓ Computed top-5 accuracy: {accuracy}%")
    
    assert abs(accuracy - expected) < 0.01, f"FAILED: Expected {expected}%, got {accuracy}%"
    print("✅ TEST PASSED!\n")


def test_edge_cases():
    """Test edge cases."""
    print("="*60)
    print("TEST 3: Edge Cases")
    print("="*60)
    
    # Test 1: All predictions correct (100%)
    predictions = torch.tensor([
        [0.1, 0.9, 0.3],  # Pred: 1, True: 1 ✓
        [0.9, 0.1, 0.2],  # Pred: 0, True: 0 ✓
        [0.1, 0.2, 0.9],  # Pred: 2, True: 2 ✓
    ])
    targets = torch.tensor([1, 0, 2])
    
    acc = compute_accuracy(predictions, targets, k=1)
    print(f"All correct: {acc}% (expected 100%)")
    assert abs(acc - 100.0) < 0.01
    
    # Test 2: All predictions wrong (0%)
    predictions = torch.tensor([
        [0.9, 0.1, 0.2],  # Pred: 0, True: 1 ✗
        [0.1, 0.9, 0.2],  # Pred: 1, True: 0 ✗
        [0.9, 0.8, 0.1],  # Pred: 0, True: 2 ✗
    ])
    targets = torch.tensor([1, 0, 2])
    
    acc = compute_accuracy(predictions, targets, k=1)
    print(f"All wrong:   {acc}% (expected 0%)")
    assert abs(acc - 0.0) < 0.01
    
    # Test 3: Single sample
    predictions = torch.tensor([[0.1, 0.9, 0.3]])
    targets = torch.tensor([1])
    
    acc = compute_accuracy(predictions, targets, k=1)
    print(f"Single sample (correct): {acc}% (expected 100%)")
    assert abs(acc - 100.0) < 0.01
    
    print("✅ ALL EDGE CASES PASSED!\n")


def test_top_k_relationship():
    """Verify that top-k accuracy increases with k (or stays the same)."""
    print("="*60)
    print("TEST 4: Top-K Relationship (k=1 ≤ k=3 ≤ k=5)")
    print("="*60)
    
    # Random predictions
    torch.manual_seed(42)
    predictions = torch.randn(100, 20)  # 100 samples, 20 classes
    targets = torch.randint(0, 20, (100,))
    
    top1 = compute_accuracy(predictions, targets, k=1)
    top3 = compute_accuracy(predictions, targets, k=3)
    top5 = compute_accuracy(predictions, targets, k=5)
    top10 = compute_accuracy(predictions, targets, k=10)
    
    print(f"Top-1:  {top1:.2f}%")
    print(f"Top-3:  {top3:.2f}%")
    print(f"Top-5:  {top5:.2f}%")
    print(f"Top-10: {top10:.2f}%")
    
    # Verify monotonic increase
    assert top1 <= top3, "Top-1 should be ≤ Top-3"
    assert top3 <= top5, "Top-3 should be ≤ Top-5"
    assert top5 <= top10, "Top-5 should be ≤ Top-10"
    
    print("\n✅ Monotonic relationship verified!")
    print("   (Top-k accuracy correctly increases with k)\n")


def test_implementation_details():
    """Test implementation matches expected PyTorch behavior."""
    print("="*60)
    print("TEST 5: Implementation Details")
    print("="*60)
    
    predictions = torch.tensor([
        [0.3, 0.1, 0.8, 0.2, 0.5],  # Top-3: [2, 4, 0]
        [0.7, 0.9, 0.2, 0.4, 0.1],  # Top-3: [1, 0, 3]
    ])
    targets = torch.tensor([4, 3])  # Sample 0: class 4, Sample 1: class 3
    
    # Manual computation for k=3
    _, top3 = predictions.topk(3, dim=-1)
    print(f"Top-3 predictions:")
    print(f"  Sample 0: {top3[0].tolist()} (true: 4)")
    print(f"  Sample 1: {top3[1].tolist()} (true: 3)")
    
    # Check if targets are in top-3
    targets_expanded = targets.unsqueeze(-1).expand_as(top3)
    print(f"\nTargets expanded: {targets_expanded.tolist()}")
    
    correct = (top3 == targets_expanded)
    print(f"Matches: {correct.tolist()}")
    
    correct_any = correct.any(dim=-1)
    print(f"Any match per sample: {correct_any.tolist()}")
    
    manual_acc = correct_any.float().mean().item() * 100
    print(f"\nManual calculation: {manual_acc}%")
    
    # Compare with function
    func_acc = compute_accuracy(predictions, targets, k=3)
    print(f"Function result:    {func_acc}%")
    
    assert abs(manual_acc - func_acc) < 0.01, "Implementation mismatch!"
    print("\n✅ Implementation matches expected behavior!\n")


if __name__ == "__main__":
    print("\n" + "🧪 TESTING ACCURACY METRIC IMPLEMENTATION" + "\n")
    
    try:
        test_top1_accuracy()
        test_top5_accuracy()
        test_edge_cases()
        test_top_k_relationship()
        test_implementation_details()
        
        print("="*60)
        print("🎉 ALL TESTS PASSED! Accuracy metrics are correct.")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
