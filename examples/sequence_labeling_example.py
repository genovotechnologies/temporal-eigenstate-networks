"""
Example: Using TEN for Sequence Labeling (Named Entity Recognition)

This example demonstrates how to use TEN_Encoder for token-level classification
tasks like NER, POS tagging, or slot filling.

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


class SequenceLabelingDataset(Dataset):
    """Dummy NER dataset for demonstration."""
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int, num_tags: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_tags = num_tags
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,))
        # Each token gets a tag (e.g., B-PER, I-PER, O, B-LOC, etc.)
        tags = torch.randint(0, self.num_tags, (self.seq_len,))
        return tokens, tags


def main():
    print("=" * 80)
    print("TEN for Sequence Labeling (NER) Example")
    print("=" * 80)
    
    # Configuration
    vocab_size = 5000
    num_tags = 9  # e.g., B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC, O
    seq_len = 128
    
    # Create TEN Encoder for sequence labeling
    print("\nCreating TEN Encoder for sequence labeling...")
    model = create_model(
        model_type='encoder',
        input_dim=vocab_size,
        d_model=256,
        num_layers=4,
        num_eigenstates=64,
        num_classes=num_tags,
        task_type='sequence_labeling',  # Per-token predictions
        use_embeddings=True,
        max_seq_len=seq_len,
        dropout=0.1,
    )
    
    print(f"✓ Model created with {model.count_parameters() / 1e6:.2f}M parameters")
    print(f"  Task: {model.task_type}")
    print(f"  Number of tags: {num_tags}")
    
    # Create datasets
    print("\nPreparing datasets...")
    train_dataset = SequenceLabelingDataset(1000, seq_len, vocab_size, num_tags)
    val_dataset = SequenceLabelingDataset(200, seq_len, vocab_size, num_tags)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Setup training
    print("\nSetting up training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Custom training loop for sequence labeling
    class SequenceLabelingTrainer(Trainer):
        def train_epoch(self, train_loader, epoch):
            self.model.train()
            total_loss = 0.0
            total_tokens = 0
            correct = 0
            
            from tqdm import tqdm
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
            
            for batch_idx, (tokens, tags) in enumerate(pbar):
                tokens = tokens.to(self.device)
                tags = tags.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                logits = self.model(tokens)  # (batch, seq_len, num_tags)
                
                # Reshape for loss
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tags.reshape(-1)
                )
                
                # Backward
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                
                # Metrics
                total_loss += loss.item()
                predictions = logits.argmax(dim=-1)
                correct += (predictions == tags).sum().item()
                total_tokens += tags.numel()
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': correct / total_tokens
                })
            
            return {
                'loss': total_loss / len(train_loader),
                'accuracy': correct / total_tokens
            }
        
        def validate(self, val_loader):
            self.model.eval()
            total_loss = 0.0
            total_tokens = 0
            correct = 0
            
            with torch.no_grad():
                for tokens, tags in val_loader:
                    tokens = tokens.to(self.device)
                    tags = tags.to(self.device)
                    
                    logits = self.model(tokens)
                    loss = self.criterion(
                        logits.reshape(-1, logits.size(-1)),
                        tags.reshape(-1)
                    )
                    
                    total_loss += loss.item()
                    predictions = logits.argmax(dim=-1)
                    correct += (predictions == tags).sum().item()
                    total_tokens += tags.numel()
            
            return {
                'val_loss': total_loss / len(val_loader),
                'val_accuracy': correct / total_tokens
            }
    
    trainer = SequenceLabelingTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        grad_clip=1.0,
    )
    
    # Train
    print("\nTraining...")
    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    for epoch in range(1, 4):
        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.validate(val_loader)
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['val_loss'])
        history['val_accuracy'].append(val_metrics['val_accuracy'])
        
        print(f"Epoch {epoch}: "
              f"Train Loss = {train_metrics['loss']:.4f}, "
              f"Train Acc = {train_metrics['accuracy']:.4f}, "
              f"Val Loss = {val_metrics['val_loss']:.4f}, "
              f"Val Acc = {val_metrics['val_accuracy']:.4f}")
    
    print(f"\n✓ Training complete!")
    print(f"  Final train accuracy: {history['train_accuracy'][-1]:.4f}")
    print(f"  Final val accuracy: {history['val_accuracy'][-1]:.4f}")
    
    # Test inference
    print("\nTesting inference...")
    model.eval()
    test_tokens = torch.randint(0, vocab_size, (1, seq_len))
    
    # Tag names for visualization
    tag_names = ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC', 'O']
    
    with torch.no_grad():
        logits = model(test_tokens)
        predictions = torch.argmax(logits, dim=-1)
    
    print(f"  Input shape: {test_tokens.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"\n  Sample predictions (first 10 tokens):")
    for i in range(10):
        token_id = test_tokens[0, i].item()
        pred_tag = predictions[0, i].item()
        print(f"    Token {token_id:4d} → {tag_names[pred_tag]}")
    
    print("\n✓ Example complete!")
    
    print("\n" + "=" * 80)
    print("Use Cases for Sequence Labeling with TEN:")
    print("=" * 80)
    print("1. Named Entity Recognition (NER)")
    print("2. Part-of-Speech (POS) Tagging")
    print("3. Slot Filling for Dialog Systems")
    print("4. Chunking and Phrase Detection")
    print("5. Event Detection in Text")
    print("6. Medical Entity Extraction")
    print("=" * 80)


if __name__ == "__main__":
    main()
