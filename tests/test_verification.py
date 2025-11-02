"""
Simple verification tests for the updated TEN implementation.
Tests all the fixed and improved components.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import (
    TemporalEigenstateNetwork,
    TemporalEigenstateConfig,
    TemporalFlowCell,
    ResonanceBlock,
    HierarchicalTEN,
    create_model,
    count_parameters,
)


def test_temporal_flow_cell():
    """Test TemporalFlowCell functionality."""
    print("Testing TemporalFlowCell...")
    
    cell = TemporalFlowCell(dim=128, num_eigenstates=32)
    x = torch.randn(2, 10, 128)
    
    # Test forward pass
    output, state = cell(x)
    assert output.shape == (2, 10, 128), f"Unexpected output shape: {output.shape}"
    
    # Test state
    assert isinstance(state, tuple) and len(state) == 2, "State should be (real, imag) tuple"
    assert state[0].shape == (2, 32), f"Unexpected state shape: {state[0].shape}"
    
    # Test with existing state
    output2, state2 = cell(x, state)
    assert output2.shape == (2, 10, 128)
    
    # Test eigenvalues
    magnitude, phase = cell.get_eigenvalues()
    assert magnitude.min() >= 0 and magnitude.max() <= 1, "Magnitudes should be in [0,1]"
    
    print("✓ TemporalFlowCell tests passed")


def test_resonance_block():
    """Test ResonanceBlock functionality."""
    print("\nTesting ResonanceBlock...")
    
    block = ResonanceBlock(dim=128, num_cells=4, num_eigenstates=64)
    x = torch.randn(2, 10, 128)
    
    # Test forward pass
    output, states = block(x)
    assert output.shape == (2, 10, 128)
    assert len(states) == 4, "Should have states for each cell"
    
    # Test with existing states
    output2, states2 = block(x, states)
    assert output2.shape == (2, 10, 128)
    
    print("✓ ResonanceBlock tests passed")


def test_ten_model():
    """Test full TEN model."""
    print("\nTesting TemporalEigenstateNetwork...")
    
    config = TemporalEigenstateConfig(
        d_model=256,
        n_heads=8,
        n_layers=4,
        num_eigenstates=64,
        vocab_size=1000,
        max_seq_len=128,
    )
    
    model = TemporalEigenstateNetwork(config)
    
    # Test with token indices
    tokens = torch.randint(0, 1000, (2, 50))
    output = model(tokens)
    assert output.shape == (2, 50, 1000), f"Unexpected output shape: {output.shape}"
    
    # Test with return_states
    output, states = model(tokens, return_states=True)
    assert len(states) == 4, "Should have states for each layer"
    
    # Test generation
    prompt = torch.randint(0, 1000, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=1.0)
    assert generated.shape == (1, 30)
    
    # Test parameter counting
    param_count = model.count_parameters()
    assert param_count > 0
    
    print(f"✓ TemporalEigenstateNetwork tests passed ({param_count/1e6:.2f}M params)")


def test_hierarchical_ten():
    """Test HierarchicalTEN model."""
    print("\nTesting HierarchicalTEN...")
    
    model = HierarchicalTEN(
        vocab_size=1000,
        dim=128,
        num_layers=2,
        scales=[1, 2, 4],
        num_eigenstates=48,
        max_seq_len=128,
    )
    
    # Test forward pass
    tokens = torch.randint(0, 1000, (2, 64))
    output = model(tokens)
    assert output.shape == (2, 64, 1000)
    
    # Test generation
    prompt = torch.randint(0, 1000, (1, 10))
    generated = model.generate(prompt, max_new_tokens=15)
    assert generated.shape == (1, 25)
    
    # Test parameter counting
    param_count = model.count_parameters()
    assert param_count > 0
    
    print(f"✓ HierarchicalTEN tests passed ({param_count/1e6:.2f}M params)")


def test_model_factory():
    """Test create_model factory function."""
    print("\nTesting model factory...")
    
    # Test TEN creation
    model1 = create_model('ten', vocab_size=1000, dim=128, num_layers=2)
    assert isinstance(model1, TemporalEigenstateNetwork)
    
    # Test HTEN creation
    model2 = create_model('hten', vocab_size=1000, dim=128, num_layers=2)
    assert isinstance(model2, HierarchicalTEN)
    
    # Test parameter counting utility
    params = count_parameters(model1)
    assert 'total' in params
    assert 'trainable' in params
    assert 'total_millions' in params
    
    print("✓ Model factory tests passed")


def test_gradient_flow():
    """Test that gradients flow properly."""
    print("\nTesting gradient flow...")
    
    config = TemporalEigenstateConfig(
        d_model=128,
        n_layers=2,
        vocab_size=500,
        max_seq_len=64,
    )
    
    model = TemporalEigenstateNetwork(config)
    tokens = torch.randint(0, 500, (2, 32))
    
    # Forward pass
    output = model(tokens)
    
    # Compute loss
    targets = torch.randint(0, 500, (2, 32))
    loss = torch.nn.functional.cross_entropy(
        output.reshape(-1, 500),
        targets.reshape(-1)
    )
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist for parameters that should have them
    # Note: Not all parameters may have gradients (e.g., if not used in forward pass)
    params_with_grads = 0
    params_total = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_total += 1
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                params_with_grads += 1
    
    # Most parameters should have gradients
    grad_ratio = params_with_grads / params_total
    assert grad_ratio > 0.5, f"Only {params_with_grads}/{params_total} parameters have gradients"
    
    print(f"✓ Gradient flow tests passed ({params_with_grads}/{params_total} params with gradients)")


def test_different_sequence_lengths():
    """Test model with various sequence lengths."""
    print("\nTesting different sequence lengths...")
    
    config = TemporalEigenstateConfig(
        d_model=128,
        n_layers=2,
        vocab_size=500,
        max_seq_len=256,
    )
    
    model = TemporalEigenstateNetwork(config)
    model.eval()
    
    with torch.no_grad():
        for seq_len in [16, 32, 64, 128, 256]:
            tokens = torch.randint(0, 500, (1, seq_len))
            output = model(tokens)
            assert output.shape == (1, seq_len, 500), f"Failed for seq_len={seq_len}"
    
    print("✓ Sequence length tests passed")


def test_numerical_stability():
    """Test numerical stability."""
    print("\nTesting numerical stability...")
    
    config = TemporalEigenstateConfig(
        d_model=128,
        n_layers=3,
        vocab_size=500,
        max_seq_len=128,
    )
    
    model = TemporalEigenstateNetwork(config)
    model.eval()
    
    # Test with many forward passes
    tokens = torch.randint(0, 500, (2, 64))
    
    with torch.no_grad():
        for _ in range(10):
            output = model(tokens)
            assert not torch.isnan(output).any(), "NaN detected in output"
            assert not torch.isinf(output).any(), "Inf detected in output"
    
    print("✓ Numerical stability tests passed")


def main():
    print("=" * 80)
    print("Running TEN Verification Tests")
    print("=" * 80)
    
    try:
        test_temporal_flow_cell()
        test_resonance_block()
        test_ten_model()
        test_hierarchical_ten()
        test_model_factory()
        test_gradient_flow()
        test_different_sequence_lengths()
        test_numerical_stability()
        
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
