"""Training script for Temporal Eigenstate Networks.

Copyright (c) 2025 Genovo Technologies. All Rights Reserved.
PROPRIETARY AND CONFIDENTIAL - Internal Use Only
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Dict, Any
import math

from .model import TemporalEigenstateNetwork, TemporalEigenstateConfig


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create a learning rate scheduler with linear warmup and cosine decay.
    
    Args:
        optimizer: The optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum learning rate as ratio of initial lr
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Trainer for Temporal Eigenstate Networks."""
    
    def __init__(
        self,
        model: TemporalEigenstateNetwork,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        language_modeling: bool = False,
        grad_clip: Optional[float] = 1.0,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: TEN model to train
            optimizer: Optimizer
            criterion: Loss function (if None, uses CrossEntropyLoss for LM)
            device: Device to use
            language_modeling: If True, expects token sequences and uses LM training
            grad_clip: Gradient clipping value (None to disable)
            scheduler: Optional learning rate scheduler
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.language_modeling = language_modeling
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        
        # Set default criterion for language modeling
        if criterion is None and language_modeling:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Handle different batch formats
            if self.language_modeling:
                # Language modeling: batch is just token sequences
                if isinstance(batch, (list, tuple)):
                    tokens = batch[0]
                else:
                    tokens = batch
                tokens = tokens.to(self.device)
                
                # Create inputs and targets
                inputs = tokens[:, :-1]  # All but last token
                targets = tokens[:, 1:]  # All but first token
            else:
                # General sequence modeling: batch is (inputs, targets)
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Compute loss
            if self.language_modeling:
                # Reshape for language modeling loss
                batch_size, seq_len, vocab_size = outputs.shape
                loss = self.criterion(
                    outputs.reshape(-1, vocab_size),
                    targets.reshape(-1)
                )
                num_tokens = targets.numel()
            else:
                loss = self.criterion(outputs, targets)
                num_tokens = 1  # Count as single sample
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            # Step scheduler if provided
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
            # Update progress bar
            current_loss = loss.item()
            if self.language_modeling:
                ppl = math.exp(current_loss)
                pbar.set_postfix({"loss": current_loss, "ppl": ppl})
            else:
                pbar.set_postfix({"loss": current_loss})
        
        avg_loss = total_loss / (total_tokens if self.language_modeling else len(train_loader))
        metrics = {"loss": avg_loss}
        if self.language_modeling:
            metrics["perplexity"] = math.exp(avg_loss)
        return metrics
    
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Handle different batch formats
                if self.language_modeling:
                    if isinstance(batch, (list, tuple)):
                        tokens = batch[0]
                    else:
                        tokens = batch
                    tokens = tokens.to(self.device)
                    
                    # Create inputs and targets
                    inputs = tokens[:, :-1]
                    targets = tokens[:, 1:]
                else:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss
                if self.language_modeling:
                    batch_size, seq_len, vocab_size = outputs.shape
                    loss = self.criterion(
                        outputs.reshape(-1, vocab_size),
                        targets.reshape(-1)
                    )
                    num_tokens = targets.numel()
                else:
                    loss = self.criterion(outputs, targets)
                    num_tokens = 1
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        avg_loss = total_loss / (total_tokens if self.language_modeling else len(val_loader))
        metrics = {"val_loss": avg_loss}
        if self.language_modeling:
            metrics["val_perplexity"] = math.exp(avg_loss)
        return metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        save_best: bool = False,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of epochs to train
            save_best: Whether to save the best model based on validation loss
            checkpoint_path: Path to save checkpoints (required if save_best=True)
            
        Returns:
            Dictionary with training history
        """
        history = {"train_loss": [], "val_loss": []}
        if self.language_modeling:
            history["train_perplexity"] = []
            history["val_perplexity"] = []
        
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            history["train_loss"].append(train_metrics["loss"])
            if self.language_modeling:
                history["train_perplexity"].append(train_metrics["perplexity"])
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history["val_loss"].append(val_metrics["val_loss"])
                
                if self.language_modeling:
                    history["val_perplexity"].append(val_metrics["val_perplexity"])
                    print(f"Epoch {epoch}: Train Loss = {train_metrics['loss']:.4f}, "
                          f"Train PPL = {train_metrics['perplexity']:.2f}, "
                          f"Val Loss = {val_metrics['val_loss']:.4f}, "
                          f"Val PPL = {val_metrics['val_perplexity']:.2f}")
                else:
                    print(f"Epoch {epoch}: Train Loss = {train_metrics['loss']:.4f}, "
                          f"Val Loss = {val_metrics['val_loss']:.4f}")
                
                # Save best model
                if save_best and val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    if checkpoint_path:
                        self.save_checkpoint(checkpoint_path, epoch, val_metrics["val_loss"])
                        print(f"  → Saved best model (val_loss={best_val_loss:.4f})")
            else:
                if self.language_modeling:
                    print(f"Epoch {epoch}: Train Loss = {train_metrics['loss']:.4f}, "
                          f"Train PPL = {train_metrics['perplexity']:.2f}")
                else:
                    print(f"Epoch {epoch}: Train Loss = {train_metrics['loss']:.4f}")
        
        return history
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        val_loss: float,
    ) -> None:
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }, path)
    
    def load_checkpoint(
        self,
        path: str,
    ) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint


if __name__ == "__main__":
    # Example usage for language modeling
    print("Example: Training TEN for Language Modeling")
    
    config = TemporalEigenstateConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_seq_len=512,
        num_eigenstates=64,
        vocab_size=10000,
        dropout=0.1,
    )
    
    model = TemporalEigenstateNetwork(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    
    # Create trainer for language modeling
    trainer = Trainer(
        model, 
        optimizer, 
        language_modeling=True,
        grad_clip=1.0,
    )
    
    print(f"Model parameters: {model.count_parameters() / 1e6:.2f}M")
    print("Trainer initialized successfully!")
    print("\nTo train, provide train_loader and call:")
    print("  history = trainer.fit(train_loader, val_loader, epochs=10)")
