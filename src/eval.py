"""Evaluation utilities for Temporal Eigenstate Networks."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm

from .model import TemporalEigenstateNetwork


class Evaluator:
    """Evaluator for Temporal Eigenstate Networks."""
    
    def __init__(
        self,
        model: TemporalEigenstateNetwork,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def evaluate(
        self,
        test_loader: DataLoader,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """Evaluate model on test set."""
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Evaluating"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        avg_loss = total_loss / len(test_loader)
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Calculate additional metrics
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        
        return {
            "loss": avg_loss,
            "mse": mse,
            "mae": mae,
        }
    
    def measure_efficiency(
        self,
        seq_lengths: List[int],
        batch_size: int = 32,
        d_model: int = 512,
        num_runs: int = 10,
    ) -> Dict[str, List[float]]:
        """Measure computational efficiency across different sequence lengths."""
        results = {
            "seq_lengths": seq_lengths,
            "forward_time": [],
            "memory_usage": [],
        }
        
        for seq_len in seq_lengths:
            # Create dummy input
            dummy_input = torch.randn(batch_size, seq_len, d_model).to(self.device)
            
            # Warm-up
            with torch.no_grad():
                _ = self.model(dummy_input)
            
            # Measure forward pass time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            import time
            times = []
            for _ in range(num_runs):
                start = time.time()
                with torch.no_grad():
                    _ = self.model(dummy_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.time() - start)
            
            avg_time = np.mean(times)
            results["forward_time"].append(avg_time)
            
            # Measure memory usage
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    _ = self.model(dummy_input)
                memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
                results["memory_usage"].append(memory)
            else:
                results["memory_usage"].append(0.0)
        
        return results
    
    def get_attention_weights(
        self,
        inputs: torch.Tensor,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Extract attention weights from the model."""
        attention_weights = []
        
        def hook_fn(module, input, output):
            if hasattr(module, 'attention_weights'):
                attention_weights.append(module.attention_weights)
        
        # Register hooks
        hooks = []
        for i, layer in enumerate(self.model.layers):
            if layer_idx is None or i == layer_idx:
                hook = layer.attention.register_forward_hook(hook_fn)
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(inputs.to(self.device))
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        if attention_weights:
            return torch.stack(attention_weights)
        return None


if __name__ == "__main__":
    from .model import TemporalEigenstateConfig
    
    # Example usage
    config = TemporalEigenstateConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_seq_len=512,
    )
    
    model = TemporalEigenstateNetwork(config)
    evaluator = Evaluator(model)
    
    # Measure efficiency
    seq_lengths = [64, 128, 256, 512, 1024]
    efficiency = evaluator.measure_efficiency(seq_lengths, batch_size=8)
    print("Efficiency measurement complete!")
    print(f"Sequence lengths: {efficiency['seq_lengths']}")
    print(f"Forward times (s): {efficiency['forward_time']}")
