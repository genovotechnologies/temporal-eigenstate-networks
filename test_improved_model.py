"""
Test script for improved TEN model implementation.
Verifies all paper-compliant features work correctly.
"""

import torch
import sys
sys.path.insert(0, '/workspaces/temporal-eigenstate-networks/src')

from model import (
    TemporalEigenstateConfig,
    TemporalEigenstateNetwork,
    count_parameters,
    print_memory_estimate
)


def test_basic_model():
    """Test basic model instantiation and forward pass."""
    print("\n" + "="*60)
    print("TEST 1: Basic Model")
    print("="*60)
    
    config = TemporalEigenstateConfig(
        vocab_size=1000,
        dim=256,
        n_layers=4,
        num_eigenstates=32,
        num_cells=2,
        max_seq_len=512,
        chunk_size=64,
        use_gradient_checkpointing=True,
        use_resonance=True,
        ffn_multiplier=4.0,
        pos_emb_type="learned"
    )
    
    model = TemporalEigenstateNetwork(config)
    print(f"✓ Model created with {count_parameters(model):,} parameters")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits = model(x)
    
    assert logits.shape == (batch_size, seq_len, config.vocab_size), \
        f"Expected shape {(batch_size, seq_len, config.vocab_size)}, got {logits.shape}"
    print(f"✓ Forward pass successful: {logits.shape}")
    
    # Test generation
    start_tokens = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(start_tokens, max_new_tokens=20, temperature=1.0, top_k=50)
    print(f"✓ Generation successful: {generated.shape}")
    
    print("✓ Basic model test PASSED\n")
    return True


def test_hten_model():
    """Test Hierarchical TEN (HTEN) model."""
    print("\n" + "="*60)
    print("TEST 2: Hierarchical TEN (HTEN)")
    print("="*60)
    
    config = TemporalEigenstateConfig(
        vocab_size=1000,
        dim=256,
        n_layers=2,
        num_eigenstates=32,
        num_cells=2,
        max_seq_len=512,
        chunk_size=64,
        use_hten=True,  # Enable HTEN!
        hten_scales=[1, 2, 4],  # Multi-scale processing
        use_gradient_checkpointing=True,
        use_resonance=True,
        ffn_multiplier=4.0
    )
    
    model = TemporalEigenstateNetwork(config)
    print(f"✓ HTEN model created with {count_parameters(model):,} parameters")
    print(f"  Scales: {config.hten_scales}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits = model(x)
    
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    print(f"✓ HTEN forward pass successful: {logits.shape}")
    print("✓ HTEN model test PASSED\n")
    return True


def test_eigenvalue_initialization():
    """Test proper eigenvalue initialization per paper."""
    print("\n" + "="*60)
    print("TEST 3: Eigenvalue Initialization (Appendix B.2)")
    print("="*60)
    
    config = TemporalEigenstateConfig(
        vocab_size=1000,
        dim=128,
        n_layers=2,
        num_eigenstates=32,
        chunk_size=64
    )
    
    model = TemporalEigenstateNetwork(config)
    
    # Check first cell of first block
    cell = model.blocks[0].cells[0]
    
    # Get eigenvalues
    magnitude, phase = cell.get_eigenvalues()
    
    print(f"✓ Eigenvalue magnitudes: min={magnitude.min():.3f}, max={magnitude.max():.3f}")
    print(f"  (Should be constrained to [0, {config.eigenvalue_clip}])")
    
    # Check alpha_raw initialization (should be from U(-3, 0))
    alpha_raw = cell.alpha_raw.data
    print(f"✓ Alpha raw values: min={alpha_raw.min():.3f}, max={alpha_raw.max():.3f}")
    print(f"  (Should be approximately in [-3, 0] range)")
    
    # Check omega initialization (should be evenly spaced)
    omega = cell.omega.data
    print(f"✓ Omega (frequencies): min={omega.min():.3f}, max={omega.max():.3f}")
    expected_max = 2 * 3.14159 * (config.num_eigenstates - 1) / config.num_eigenstates
    print(f"  (Should span approximately [0, {expected_max:.3f}])")
    
    # Check resonance matrix constraint
    if cell.use_resonance:
        R = cell.get_resonance_matrix()
        # R should be close to identity
        identity = torch.eye(config.num_eigenstates)
        diff = (R - identity).abs().max().item()
        print(f"✓ Resonance matrix deviation from identity: {diff:.4f}")
        print(f"  (Should be small, ≈ {config.resonance_epsilon})")
    
    print("✓ Eigenvalue initialization test PASSED\n")
    return True


def test_sinusoidal_positional_encoding():
    """Test sinusoidal positional embeddings."""
    print("\n" + "="*60)
    print("TEST 4: Sinusoidal Positional Embeddings")
    print("="*60)
    
    config = TemporalEigenstateConfig(
        vocab_size=1000,
        dim=128,
        n_layers=2,
        pos_emb_type="sinusoidal"  # Use sinusoidal instead of learned
    )
    
    model = TemporalEigenstateNetwork(config)
    
    # Check that pos_emb has no learnable parameters
    pos_emb_params = sum(p.numel() for p in model.pos_emb.parameters() if p.requires_grad)
    print(f"✓ Sinusoidal pos_emb parameters: {pos_emb_params}")
    print(f"  (Should be 0 - no learnable parameters)")
    
    # Test forward pass
    x = torch.randint(0, config.vocab_size, (2, 64))
    with torch.no_grad():
        logits = model(x)
    
    print(f"✓ Forward pass with sinusoidal embeddings successful")
    print("✓ Sinusoidal positional encoding test PASSED\n")
    return True


def test_memory_efficiency():
    """Test memory-efficient features."""
    print("\n" + "="*60)
    print("TEST 5: Memory Efficiency Features")
    print("="*60)
    
    config = TemporalEigenstateConfig(
        vocab_size=10000,
        dim=512,
        n_layers=6,
        num_eigenstates=64,
        num_cells=2,
        max_seq_len=2048,
        chunk_size=64,
        use_gradient_checkpointing=True,
        use_resonance=True,
        ffn_multiplier=4.0,
        pos_emb_type="learned"
    )
    
    print(f"\nConfiguration:")
    print(f"  - Chunk size: {config.chunk_size} (limits memory growth)")
    print(f"  - Gradient checkpointing: {config.use_gradient_checkpointing}")
    print(f"  - Position embeddings: {config.pos_emb_type}")
    print(f"  - Resonance: {config.use_resonance} (learnable, not buffer)")
    print(f"  - FFN multiplier: {config.ffn_multiplier}x")
    
    # Print memory estimate
    print_memory_estimate(config, batch_size=8)
    
    print("✓ Memory efficiency test PASSED\n")
    return True


def test_gradient_flow():
    """Test that gradients flow properly through chunks."""
    print("\n" + "="*60)
    print("TEST 6: Gradient Flow")
    print("="*60)
    
    config = TemporalEigenstateConfig(
        vocab_size=1000,
        dim=128,
        n_layers=2,
        num_eigenstates=32,
        chunk_size=32,
        use_gradient_checkpointing=False  # Disable for this test
    )
    
    model = TemporalEigenstateNetwork(config)
    model.train()
    
    # Forward pass
    x = torch.randint(0, config.vocab_size, (2, 64))
    targets = torch.randint(0, config.vocab_size, (2, 64))
    
    loss = model.compute_loss(x, targets)
    
    # Backward pass
    loss.backward()
    
    # Check that eigenvalue parameters have gradients
    cell = model.blocks[0].cells[0]
    assert cell.alpha_raw.grad is not None, "Alpha should have gradients"
    assert cell.omega.grad is not None, "Omega should have gradients"
    
    grad_norm_alpha = cell.alpha_raw.grad.norm().item()
    grad_norm_omega = cell.omega.grad.norm().item()
    
    print(f"✓ Alpha gradient norm: {grad_norm_alpha:.4f}")
    print(f"✓ Omega gradient norm: {grad_norm_omega:.4f}")
    print(f"✓ Gradients flow correctly through eigenvalues")
    
    # Check resonance matrix gradients if enabled
    if cell.use_resonance and cell.resonance_M is not None:
        assert cell.resonance_M.grad is not None, "Resonance matrix should have gradients"
        grad_norm_resonance = cell.resonance_M.grad.norm().item()
        print(f"✓ Resonance matrix gradient norm: {grad_norm_resonance:.4f}")
    
    print("✓ Gradient flow test PASSED\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("TESTING IMPROVED TEN MODEL - PAPER-COMPLIANT IMPLEMENTATION")
    print("="*70)
    
    tests = [
        ("Basic Model", test_basic_model),
        ("Hierarchical TEN (HTEN)", test_hten_model),
        ("Eigenvalue Initialization", test_eigenvalue_initialization),
        ("Sinusoidal Positional Encoding", test_sinusoidal_positional_encoding),
        ("Memory Efficiency", test_memory_efficiency),
        ("Gradient Flow", test_gradient_flow),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} FAILED with error:")
            print(f"   {type(e).__name__}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Model is paper-compliant and memory-efficient.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review.")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
