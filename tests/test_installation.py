"""
Test script to verify TEN package installation and basic functionality.
"""
import sys

def test_installation():
    """Test that the package is correctly installed and functional."""
    
    print("=" * 60)
    print("TEN Package Installation Test")
    print("=" * 60)
    
    # Test 1: Import the package
    print("\n1. Testing package import...")
    try:
        from src import TemporalEigenstateNetwork, TemporalEigenstateConfig
        from src import __version__
        print(f"   ✓ Successfully imported TEN v{__version__}")
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        return False
    
    # Test 2: Import submodules
    print("\n2. Testing submodule imports...")
    try:
        from src.model import TemporalEigenstateNetwork, TemporalEigenstateConfig
        from src.train import Trainer
        from src.eval import Evaluator
        print("   ✓ All submodules imported successfully")
    except ImportError as e:
        print(f"   ✗ Submodule import failed: {e}")
        return False
    
    # Test 3: Create a model configuration
    print("\n3. Testing model configuration...")
    try:
        config = TemporalEigenstateConfig(
            d_model=256,
            n_heads=4,
            n_layers=2,
            d_ff=1024,
            max_seq_len=128,
            num_eigenstates=32,
        )
        print(f"   ✓ Configuration created: d_model={config.d_model}, n_layers={config.n_layers}")
    except Exception as e:
        print(f"   ✗ Configuration failed: {e}")
        return False
    
    # Test 4: Create a model instance
    print("\n4. Testing model instantiation...")
    try:
        model = TemporalEigenstateNetwork(config)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"   ✓ Model created with {num_params:,} parameters")
    except Exception as e:
        print(f"   ✗ Model instantiation failed: {e}")
        return False
    
    # Test 5: Forward pass
    print("\n5. Testing forward pass...")
    try:
        import torch
        batch_size, seq_len = 2, 32
        x = torch.randn(batch_size, seq_len, config.d_model)
        
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, seq_len, config.d_model), \
            f"Output shape mismatch: expected {x.shape}, got {output.shape}"
        
        print(f"   ✓ Forward pass successful: input {x.shape} → output {output.shape}")
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        return False
    
    # Test 6: Gradient computation
    print("\n6. Testing gradient computation...")
    try:
        model.train()
        x = torch.randn(2, 16, config.d_model, requires_grad=True)
        output = model(x)
        loss = output.mean()
        loss.backward()
        
        # Check gradients exist
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad, "No gradients computed"
        assert x.grad is not None, "Input gradients not computed"
        
        print("   ✓ Gradients computed successfully")
    except Exception as e:
        print(f"   ✗ Gradient computation failed: {e}")
        return False
    
    # Test 7: Test Trainer class
    print("\n7. Testing Trainer class...")
    try:
        import torch.nn as nn
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        trainer = Trainer(model, optimizer, criterion, device='cpu')
        print("   ✓ Trainer initialized successfully")
    except Exception as e:
        print(f"   ✗ Trainer initialization failed: {e}")
        return False
    
    # Test 8: Test Evaluator class
    print("\n8. Testing Evaluator class...")
    try:
        evaluator = Evaluator(model, device='cpu')
        print("   ✓ Evaluator initialized successfully")
    except Exception as e:
        print(f"   ✗ Evaluator initialization failed: {e}")
        return False
    
    # Test 9: Multiple model sizes
    print("\n9. Testing different model sizes...")
    try:
        sizes = [
            (128, 4, 2),  # small
            (256, 8, 4),  # medium
            (512, 8, 6),  # large
        ]
        for d_model, n_heads, n_layers in sizes:
            cfg = TemporalEigenstateConfig(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
            )
            mdl = TemporalEigenstateNetwork(cfg)
            x = torch.randn(1, 10, d_model)
            with torch.no_grad():
                out = mdl(x)
            assert out.shape == x.shape
        print("   ✓ Multiple model sizes work correctly")
    except Exception as e:
        print(f"   ✗ Multiple sizes test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe TEN package is correctly installed and functional.")
    print("\nYou can now use it in your applications:")
    print("  from src import TemporalEigenstateNetwork, TemporalEigenstateConfig")
    print("\nFor more examples, see INSTALLATION.md")
    print("=" * 60)
    
    return True


def usage_example():
    """Show a quick usage example."""
    print("\n" + "=" * 60)
    print("Quick Usage Example")
    print("=" * 60)
    
    code = '''
from src import TemporalEigenstateNetwork, TemporalEigenstateConfig
import torch

# Configure model
config = TemporalEigenstateConfig(
    d_model=512,
    n_heads=8,
    n_layers=6,
    max_seq_len=2048,
)

# Create model
model = TemporalEigenstateNetwork(config)

# Use in your application
x = torch.randn(4, 100, 512)  # (batch, seq_len, d_model)
output = model(x)

print(f"Input: {x.shape} → Output: {output.shape}")
'''
    
    print(code)
    print("=" * 60)


if __name__ == "__main__":
    success = test_installation()
    
    if success:
        usage_example()
        sys.exit(0)
    else:
        print("\n❌ Installation verification failed!")
        print("Please check the error messages above.")
        sys.exit(1)
