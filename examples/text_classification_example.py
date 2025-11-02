"""
Example: Using TEN for Text Classification

This example demonstrates how to use TEN_Encoder for sentiment analysis,
topic classification, or any text classification task.

Copyright (c) 2025 Genovo Technologies. All Rights Reserved.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import create_model
from src.train import Trainer


class TextClassificationDataset(Dataset):
    """Dummy dataset for demonstration."""
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int, num_classes: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,))
        label = torch.randint(0, self.num_classes, (1,)).item()
        return tokens, label


def main():
    print("=" * 80)
    print("TEN for Text Classification Example")
    print("=" * 80)
    
    # Configuration
    vocab_size = 5000
    num_classes = 3  # e.g., positive, negative, neutral
    seq_len = 128
    
    # Create TEN Encoder for classification
    print("\nCreating TEN Encoder...")
    model = create_model(
        model_type='encoder',
        input_dim=vocab_size,
        d_model=256,
        num_layers=4,
        num_eigenstates=64,
        num_classes=num_classes,
        task_type='classification',
        pooling='mean',  # Options: 'mean', 'max', 'last', 'cls'
        use_embeddings=True,  # Use embeddings for discrete tokens
        max_seq_len=seq_len,
        dropout=0.1,
    )
    
    print(f"✓ Model created with {model.count_parameters() / 1e6:.2f}M parameters")
    print(f"  Task: {model.task_type}")
    print(f"  Pooling: {model.pooling}")
    print(f"  Output classes: {num_classes}")
    
    # Create datasets
    print("\nPreparing datasets...")
    train_dataset = TextClassificationDataset(1000, seq_len, vocab_size, num_classes)
    val_dataset = TextClassificationDataset(200, seq_len, vocab_size, num_classes)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Setup training
    print("\nSetting up training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        language_modeling=False,  # Not language modeling
        grad_clip=1.0,
    )
    
    # Train
    print("\nTraining...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
    )
    
    print(f"\n✓ Training complete!")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    
    # Test inference
    print("\nTesting inference...")
    model.eval()
    test_tokens = torch.randint(0, vocab_size, (1, seq_len))
    
    with torch.no_grad():
        logits = model(test_tokens)
        probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
    
    print(f"  Input shape: {test_tokens.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Predicted class: {predicted_class}")
    print(f"  Confidence: {probs[0, predicted_class]:.4f}")
    
    print("\n✓ Example complete!")


if __name__ == "__main__":
    main()
