"""Unit tests for Temporal Eigenstate Networks."""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import (
    TemporalEigenstateNetwork,
    TemporalEigenstateConfig,
    EigenstateAttention,
)


class TestTemporalEigenstateConfig:
    """Test configuration class."""
    
    def test_default_config(self):
        config = TemporalEigenstateConfig()
        assert config.d_model == 512
        assert config.n_heads == 8
        assert config.n_layers == 6
    
    def test_custom_config(self):
        config = TemporalEigenstateConfig(
            d_model=256,
            n_heads=4,
            n_layers=3,
        )
        assert config.d_model == 256
        assert config.n_heads == 4
        assert config.n_layers == 3


class TestEigenstateAttention:
    """Test eigenstate attention mechanism."""
    
    def test_initialization(self):
        attention = EigenstateAttention(d_model=512, n_heads=8)
        assert attention.d_model == 512
        assert attention.n_heads == 8
        assert attention.d_k == 64
    
    def test_forward_pass(self):
        batch_size, seq_len, d_model = 2, 10, 512
        attention = EigenstateAttention(d_model=d_model, n_heads=8)
        
        x = torch.randn(batch_size, seq_len, d_model)
        output = attention(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_eigenspace_projection(self):
        batch_size, seq_len, d_model = 2, 10, 512
        attention = EigenstateAttention(d_model=d_model, n_heads=8)
        
        x = torch.randn(batch_size, seq_len, d_model)
        output = attention(x)
        
        # Check output is not NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestTemporalEigenstateNetwork:
    """Test the full TEN model."""
    
    def test_initialization(self):
        config = TemporalEigenstateConfig(
            d_model=512,
            n_heads=8,
            n_layers=6,
        )
        model = TemporalEigenstateNetwork(config)
        assert len(model.layers) == 6
    
    def test_forward_pass(self):
        config = TemporalEigenstateConfig(
            d_model=512,
            n_heads=8,
            n_layers=6,
            max_seq_len=128,
        )
        model = TemporalEigenstateNetwork(config)
        
        batch_size, seq_len = 2, 64
        x = torch.randn(batch_size, seq_len, 512)
        
        output = model(x)
        assert output.shape == (batch_size, seq_len, 512)
    
    def test_different_sequence_lengths(self):
        config = TemporalEigenstateConfig(
            d_model=256,
            n_heads=4,
            n_layers=3,
            max_seq_len=512,
        )
        model = TemporalEigenstateNetwork(config)
        
        for seq_len in [32, 64, 128, 256]:
            x = torch.randn(1, seq_len, 256)
            output = model(x)
            assert output.shape == (1, seq_len, 256)
    
    def test_gradient_flow(self):
        config = TemporalEigenstateConfig(
            d_model=128,
            n_heads=4,
            n_layers=2,
        )
        model = TemporalEigenstateNetwork(config)
        
        x = torch.randn(2, 10, 128, requires_grad=True)
        output = model(x)
        loss = output.mean()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestModelEfficiency:
    """Test computational efficiency."""
    
    def test_linear_complexity(self):
        """Test that complexity scales better than O(n^2)."""
        config = TemporalEigenstateConfig(
            d_model=256,
            n_heads=4,
            n_layers=2,
        )
        model = TemporalEigenstateNetwork(config)
        model.eval()
        
        import time
        times = []
        seq_lengths = [32, 64, 128, 256]
        
        with torch.no_grad():
            for seq_len in seq_lengths:
                x = torch.randn(1, seq_len, 256)
                
                # Warm-up
                _ = model(x)
                
                # Measure
                start = time.time()
                for _ in range(10):
                    _ = model(x)
                elapsed = time.time() - start
                times.append(elapsed / 10)
        
        # Check that time doesn't grow quadratically
        # For quadratic: time[3]/time[0] should be ~64
        # For linear: time[3]/time[0] should be ~8
        ratio = times[-1] / times[0]
        assert ratio < 20, f"Complexity ratio too high: {ratio}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
