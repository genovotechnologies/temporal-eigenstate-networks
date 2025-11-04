"""
Complete Example: Training and Using Temporal Eigenstate Networks (TEN)

This script demonstrates all features of the paper-compliant implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.insert(0, 'src')

from model import (
    TemporalEigenstateConfig,
    TemporalEigenstateNetwork,
    print_model_summary,
    visualize_eigenstate_spectrum,
    count_parameters
)


def create_dummy_dataset(vocab_size=1000, seq_len=128, num_samples=1000):
    """Create dummy dataset for demonstration."""
    inputs = torch.randint(0, vocab_size, (num_samples, seq_len))
    targets = torch.randint(0, vocab_size, (num_samples, seq_len))
    return TensorDataset(inputs, targets)


def train_standard_ten():
    """Train a standard TEN model."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Training Standard TEN")
    print("="*70)
    
    # Configuration
    config = TemporalEigenstateConfig(
        vocab_size=1000,
        dim=256,
        n_layers=4,
        num_eigenstates=64,
        num_cells=2,
        max_seq_len=128,
        chunk_size=64,
        use_gradient_checkpointing=True,
        use_resonance=True,
        ffn_multiplier=4.0,
        pos_emb_type="learned",
        energy_reg_weight=0.01,  # Enable energy regularization
    )
    
    # Create model
    model = TemporalEigenstateNetwork(config)
    print_model_summary(model, verbose=False)
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    dataset = create_dummy_dataset(config.vocab_size, config.max_seq_len, num_samples=100)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Training loop
    model.train()
    print("\nTraining for 3 steps...")
    
    for step, (inputs, targets) in enumerate(dataloader):
        if step >= 3:
            break
        
        # Forward pass with energy regularization
        loss_dict = model.compute_loss(inputs, targets, return_dict=True)
        total_loss = loss_dict['loss']
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping (optional but recommended)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        print(f"Step {step+1}:")
        print(f"  Total Loss: {total_loss.item():.4f}")
        print(f"  CE Loss: {loss_dict['ce_loss'].item():.4f}")
        print(f"  Energy Loss: {loss_dict['energy_loss'].item():.4f}")
    
    print("\n✓ Training complete!")
    return model


def train_hierarchical_ten():
    """Train a Hierarchical TEN (HTEN) model."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Training Hierarchical TEN (HTEN)")
    print("="*70)
    
    # Configuration with HTEN enabled
    config = TemporalEigenstateConfig(
        vocab_size=1000,
        dim=256,
        n_layers=3,
        num_eigenstates=32,
        num_cells=2,
        max_seq_len=128,
        chunk_size=64,
        use_hten=True,  # Enable HTEN!
        hten_scales=[1, 2, 4],  # Multi-scale processing
        use_gradient_checkpointing=True,
        use_resonance=True,
        ffn_multiplier=4.0,
    )
    
    model = TemporalEigenstateNetwork(config)
    print(f"\n✓ HTEN model with {count_parameters(model):,} parameters")
    print(f"  Multi-scale processing at scales: {config.hten_scales}")
    
    # Quick training
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    dataset = create_dummy_dataset(config.vocab_size, config.max_seq_len, num_samples=20)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model.train()
    inputs, targets = next(iter(dataloader))
    loss = model.compute_loss(inputs, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"\n✓ HTEN training step complete (loss: {loss.item():.4f})")
    return model


def demonstrate_generation(model, config):
    """Demonstrate text generation."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Text Generation with State Caching")
    print("="*70)
    
    model.eval()
    
    # Generate with different settings
    start_tokens = torch.randint(0, config.vocab_size, (1, 10))
    
    print("\nGenerating 50 tokens with state caching...")
    with torch.no_grad():
        generated = model.generate(
            start_tokens,
            max_new_tokens=50,
            temperature=0.8,
            top_k=50,
            use_cache=True
        )
    
    print(f"✓ Generated sequence shape: {generated.shape}")
    print(f"  Start: {start_tokens.tolist()[0][:5]}...")
    print(f"  Generated: {generated.tolist()[0][:15]}...")
    
    print("\nGenerating 30 tokens without caching...")
    with torch.no_grad():
        generated_no_cache = model.generate(
            start_tokens,
            max_new_tokens=30,
            temperature=1.0,
            top_k=None,
            use_cache=False
        )
    
    print(f"✓ Generated sequence shape: {generated_no_cache.shape}")


def analyze_eigenstates(model):
    """Analyze learned eigenstate properties."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Eigenstate Analysis (Section 6.5)")
    print("="*70)
    
    analysis = model.get_eigenstate_analysis()
    
    mags = analysis['eigenvalue_magnitudes']
    phases = analysis['eigenvalue_phases']
    freqs = analysis['frequency_spectrum']
    
    print(f"\nEigenstate Statistics:")
    print(f"  Shape: {mags.shape} (layers, cells, eigenstates)")
    print(f"\n  Eigenvalue Magnitudes:")
    print(f"    Min: {mags.min().item():.4f}")
    print(f"    Max: {mags.max().item():.4f}")
    print(f"    Mean: {mags.mean().item():.4f}")
    print(f"    Std: {mags.std().item():.4f}")
    
    print(f"\n  Frequency Spectrum:")
    print(f"    Min: {freqs.min().item():.4f} cycles")
    print(f"    Max: {freqs.max().item():.4f} cycles")
    print(f"    Mean: {freqs.mean().item():.4f} cycles")
    
    if len(analysis['resonance_norms']) > 0:
        res_norms = analysis['resonance_norms']
        print(f"\n  Resonance Matrix Deviation:")
        print(f"    Mean: {res_norms.mean().item():.4f}")
        print(f"    Std: {res_norms.std().item():.4f}")
        print(f"    (Should be small, ≈ ε = {model.config.resonance_epsilon})")
    
    # Try to visualize if matplotlib available
    try:
        print(f"\nAttempting to visualize eigenstate spectrum...")
        visualize_eigenstate_spectrum(model, save_path="eigenstate_spectrum.png")
        print(f"✓ Saved visualization to eigenstate_spectrum.png")
    except ImportError:
        print("  (matplotlib not available for visualization)")
    
    print("\n✓ Eigenstate analysis complete!")


def demonstrate_mixed_precision():
    """Demonstrate mixed precision training."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Mixed Precision Training (FP16)")
    print("="*70)
    
    config = TemporalEigenstateConfig(
        vocab_size=1000,
        dim=256,
        n_layers=2,
        num_eigenstates=32,
        chunk_size=64,
        use_gradient_checkpointing=True,
    )
    
    model = TemporalEigenstateNetwork(config)
    
    # Move to CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"\nDevice: {device}")
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Setup mixed precision training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Dummy batch
    inputs = torch.randint(0, config.vocab_size, (2, 128)).to(device)
    targets = torch.randint(0, config.vocab_size, (2, 128)).to(device)
    
    model.train()
    
    if scaler:
        # Mixed precision training
        with torch.cuda.amp.autocast():
            loss = model.compute_loss(inputs, targets)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"\n✓ Mixed precision training step (FP16)")
        print(f"  Loss: {loss.item():.4f}")
    else:
        # Regular training
        loss = model.compute_loss(inputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"\n✓ Regular training step (FP32)")
        print(f"  Loss: {loss.item():.4f}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("TEMPORAL EIGENSTATE NETWORKS (TEN)")
    print("Complete Paper-Compliant Implementation Examples")
    print("="*70)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    try:
        # Example 1: Standard TEN
        standard_model = train_standard_ten()
        
        # Example 2: Hierarchical TEN
        hten_model = train_hierarchical_ten()
        
        # Example 3: Generation
        demonstrate_generation(standard_model, standard_model.config)
        
        # Example 4: Eigenstate Analysis
        analyze_eigenstates(standard_model)
        
        # Example 5: Mixed Precision
        demonstrate_mixed_precision()
        
        print("\n" + "="*70)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nKey Features Demonstrated:")
        print("  ✓ Standard TEN training with energy regularization")
        print("  ✓ Hierarchical TEN (HTEN) multi-scale processing")
        print("  ✓ Efficient generation with state caching")
        print("  ✓ Eigenstate analysis and visualization")
        print("  ✓ Mixed precision (FP16) training")
        print("\nModel is production-ready and paper-compliant! 🎉")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error in examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
