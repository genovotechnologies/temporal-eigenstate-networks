"""Training script for Temporal Eigenstate Networks."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Dict, Any

from .model import TemporalEigenstateNetwork, TemporalEigenstateConfig


class Trainer:
    """Trainer for Temporal Eigenstate Networks."""
    
    def __init__(
        self,
        model: TemporalEigenstateNetwork,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        return {"loss": avg_loss}
    
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return {"val_loss": avg_loss}
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
    ) -> Dict[str, Any]:
        """Train the model for multiple epochs."""
        history = {"train_loss": [], "val_loss": []}
        
        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            history["train_loss"].append(train_metrics["loss"])
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history["val_loss"].append(val_metrics["val_loss"])
                print(f"Epoch {epoch}: Train Loss = {train_metrics['loss']:.4f}, "
                      f"Val Loss = {val_metrics['val_loss']:.4f}")
            else:
                print(f"Epoch {epoch}: Train Loss = {train_metrics['loss']:.4f}")
        
        return history


if __name__ == "__main__":
    # Example usage
    config = TemporalEigenstateConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_seq_len=512,
    )
    
    model = TemporalEigenstateNetwork(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    trainer = Trainer(model, optimizer, criterion)
    print("Trainer initialized successfully!")
