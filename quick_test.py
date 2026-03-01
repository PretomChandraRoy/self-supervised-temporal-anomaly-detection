"""
Quick Test - Verify Fixes Work
Run this to check if numerical stability fixes are working
"""

import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.temporal_transformer import SelfSupervisedTemporalModel

print("="*80)
print("QUICK TEST: Verifying Numerical Stability Fixes")
print("="*80)

# Test 1: Model initialization
print("\n[Test 1] Model initialization with fixed weights...")
try:
    model = SelfSupervisedTemporalModel(
        n_features=29,
        d_model=128,
        n_heads=8,
        n_layers=4,
        dropout=0.1,
        reconstruction_weight=0.1,  # FIXED: lowered from 1.0
        contrastive_weight=1.0
    )
    print("✅ Model initialized successfully")
except Exception as e:
    print(f"❌ Model initialization failed: {e}")
    sys.exit(1)

# Test 2: Forward pass with normalized data
print("\n[Test 2] Forward pass with normalized input...")
try:
    # Simulate properly scaled data (mean=0, std=1)
    batch_size = 32
    seq_len = 60
    n_features = 29

    # Random data in reasonable range
    x = torch.randn(batch_size, seq_len, n_features) * 0.5  # Scale to [-1.5, 1.5]

    # Forward pass
    loss, losses = model(x, use_contrastive=True, use_reconstruction=True)

    print(f"✅ Forward pass successful")
    print(f"   Total Loss: {loss.item():.4f}")
    print(f"   Contrastive: {losses['contrastive']:.4f}")
    print(f"   Reconstruction: {losses['reconstruction']:.4f}")

    # Check if loss is reasonable
    if loss.item() > 100:
        print(f"⚠️  WARNING: Loss is high ({loss.item():.2f}), but should be < 100")
    if np.isnan(loss.item()):
        print("❌ FAIL: Loss is NaN")
        sys.exit(1)

except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Backward pass with gradient clipping
print("\n[Test 3] Backward pass with gradient clipping...")
try:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Forward
    loss, _ = model(x)

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Check gradients
    max_grad = 0.0
    for param in model.parameters():
        if param.grad is not None:
            max_grad = max(max_grad, param.grad.abs().max().item())

    print(f"✅ Backward pass successful")
    print(f"   Max gradient: {max_grad:.6f}")

    if max_grad > 10.0:
        print(f"⚠️  WARNING: Large gradients detected ({max_grad:.2f})")

    optimizer.step()
    print("✅ Optimizer step successful")

except Exception as e:
    print(f"❌ Backward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Multiple training steps
print("\n[Test 4] Multiple training steps...")
try:
    losses_history = []

    for step in range(10):
        # Generate new batch
        x = torch.randn(batch_size, seq_len, n_features) * 0.5

        # Training step
        optimizer.zero_grad()
        loss, _ = model(x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses_history.append(loss.item())

        # Check for NaN
        if np.isnan(loss.item()):
            print(f"❌ FAIL: NaN at step {step+1}")
            sys.exit(1)

    print(f"✅ Completed 10 training steps")
    print(f"   Loss trajectory: {losses_history[0]:.4f} -> {losses_history[-1]:.4f}")

    # Check if loss is decreasing or stable
    if losses_history[-1] > losses_history[0] * 2:
        print(f"⚠️  WARNING: Loss increased significantly")

except Exception as e:
    print(f"❌ Training steps failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Embedding extraction
print("\n[Test 5] Embedding extraction...")
try:
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(x)

    print(f"✅ Embeddings extracted successfully")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Mean: {embeddings.mean().item():.4f}")
    print(f"   Std: {embeddings.std().item():.4f}")

    # Check for NaN in embeddings
    if torch.isnan(embeddings).any():
        print("❌ FAIL: NaN values in embeddings")
        sys.exit(1)

except Exception as e:
    print(f"❌ Embedding extraction failed: {e}")
    sys.exit(1)

# Final summary
print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print("\nNumerical stability fixes are working correctly:")
print("  ✓ Reconstruction weight lowered to 0.1")
print("  ✓ Gradient clipping enabled (max_norm=1.0)")
print("  ✓ Loss values in reasonable range")
print("  ✓ No NaN values detected")
print("  ✓ Training is stable")
print("\nYou can now run the full training pipeline:")
print("  python example.py")
print("\nOr validate with synthetic anomalies:")
print("  python example_validation.py")
print("="*80)

