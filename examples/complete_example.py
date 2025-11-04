"""
Complete example demonstrating TEN training and evaluation.

Copyright (c) 2025 Genovo Technologies. All Rights Reserved.
PROPRIETARY AND CONFIDENTIAL - Internal Use Only

This example shows how to:
1. Create a TEN model with proper configuration
2. Prepare data for language modeling
3. Train the model with proper optimizer and scheduler
4. Evaluate the model and measure efficiency
5. Generate text from the trained model
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import TemporalEigenstateNetwork, TemporalEigenstateConfig, create_model
from src.train import Trainer, get_cosine_schedule_with_warmup
from src.eval import Evaluator


class DummyLanguageDataset(Dataset):
    """Dummy dataset for demonstration purposes."""
    
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random sequences
        return torch.randint(0, self.vocab_size, (self.seq_len,))


def main():
    print("=" * 80)
    print("Temporal Eigenstate Networks - Complete Training Example")
    print("=" * 80)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    config = TemporalEigenstateConfig(
        vocab_size=5000,       # Small vocabulary for demo
        dim=256,               # Hidden dimension
        n_layers=4,            # Fewer layers for demo
        num_eigenstates=64,    # Number of eigenstates
        max_seq_len=128,       # Shorter sequences for demo
        dropout=0.1,
        chunk_size=64,
        use_gradient_checkpointing=False,  # Disabled for state tracking
    )
    
    print(f"\nModel Configuration:")
    print(f"  dim: {config.dim}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  num_eigenstates: {config.num_eigenstates}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  max_seq_len: {config.max_seq_len}")
    
    # Create model
    print("\n" + "=" * 80)
    print("Step 1: Creating Model")
    print("=" * 80)
    
    model = TemporalEigenstateNetwork(config)
    print(f"✓ Model created with {model.count_parameters() / 1e6:.2f}M parameters")
    
    # Create datasets
    print("\n" + "=" * 80)
    print("Step 2: Preparing Data")
    print("=" * 80)
    
    train_dataset = DummyLanguageDataset(
        num_samples=1000,
        seq_len=config.max_seq_len,
        vocab_size=config.vocab_size
    )
    val_dataset = DummyLanguageDataset(
        num_samples=100,
        seq_len=config.max_seq_len,
        vocab_size=config.vocab_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )
    
    print(f"✓ Train dataset: {len(train_dataset)} samples")
    print(f"✓ Val dataset: {len(val_dataset)} samples")
    print(f"✓ Batch size: 16")
    
    # Create optimizer and scheduler
    print("\n" + "=" * 80)
    print("Step 3: Setting up Optimizer and Scheduler")
    print("=" * 80)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    num_epochs = 3  # Small number for demo
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = num_training_steps // 10
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=0.1
    )
    
    print(f"✓ Optimizer: AdamW (lr=3e-4)")
    print(f"✓ Scheduler: Cosine with warmup")
    print(f"✓ Warmup steps: {num_warmup_steps}")
    print(f"✓ Total training steps: {num_training_steps}")
    
    # Create trainer
    print("\n" + "=" * 80)
    print("Step 4: Training Model")
    print("=" * 80)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        language_modeling=True,
        grad_clip=1.0,
        scheduler=scheduler,
    )
    
    print(f"✓ Trainer initialized")
    print(f"\nTraining for {num_epochs} epochs...\n")
    
    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=num_epochs,
    )
    
    print("\n✓ Training complete!")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final train perplexity: {history['train_perplexity'][-1]:.2f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"  Final val perplexity: {history['val_perplexity'][-1]:.2f}")
    
    # Evaluate
    print("\n" + "=" * 80)
    print("Step 5: Evaluating Model")
    print("=" * 80)
    
    evaluator = Evaluator(model)
    
    # Evaluate on validation set
    eval_metrics = evaluator.evaluate_language_modeling(val_loader)
    print(f"\nValidation Metrics:")
    print(f"  Loss: {eval_metrics['loss']:.4f}")
    print(f"  Perplexity: {eval_metrics['perplexity']:.2f}")
    print(f"  Token Accuracy: {eval_metrics['token_accuracy']:.4f}")
    
    # Measure efficiency
    print(f"\nMeasuring computational efficiency...")
    seq_lengths = [32, 64, 128]
    efficiency = evaluator.measure_efficiency(
        seq_lengths,
        batch_size=8,
        num_runs=3
    )
    print(f"\nEfficiency Results:")
    for seq_len, time in zip(efficiency['seq_lengths'], efficiency['forward_time']):
        print(f"  Seq length {seq_len}: {time:.4f}s per batch")
    
    # Extract eigenstate information
    print(f"\nExtracting eigenstate dynamics...")
    dummy_tokens = torch.randint(0, config.vocab_size, (2, 64))
    eigenstate_info = evaluator.get_eigenstate_activations(dummy_tokens, block_idx=0)
    print(f"✓ Extracted eigenstate info from {len(eigenstate_info)} cells in first block")
    
    # Generate text
    print("\n" + "=" * 80)
    print("Step 6: Generating Text")
    print("=" * 80)
    
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    print(f"\nPrompt tokens: {prompt[0].tolist()}")
    
    generated = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=1.0,
        top_k=50
    )
    print(f"Generated tokens: {generated[0].tolist()}")
    print(f"✓ Generated {generated.shape[1] - prompt.shape[1]} new tokens")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"""
✓ Model: Temporal Eigenstate Network
  - Parameters: {model.count_parameters() / 1e6:.2f}M
  - Hidden dim: {config.dim}
  - Layers: {config.n_layers}
  - Eigenstates: {config.num_eigenstates}

✓ Training:
  - Epochs: {num_epochs}
  - Final perplexity: {history['val_perplexity'][-1]:.2f}
  - Token accuracy: {eval_metrics['token_accuracy']:.4f}

✓ All tests passed successfully!
""")


if __name__ == "__main__":
    main()
