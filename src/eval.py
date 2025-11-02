"""Evaluation utilities for Temporal Eigenstate Networks.

Copyright (c) 2025 Genovo Technologies. All Rights Reserved.
PROPRIETARY AND CONFIDENTIAL - Internal Use Only
"""

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
        num_runs: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Measure computational efficiency across different sequence lengths.
        
        Args:
            seq_lengths: List of sequence lengths to test
            batch_size: Batch size for testing
            num_runs: Number of runs for averaging
            
        Returns:
            Dictionary with timing and memory measurements
        """
        results = {
            "seq_lengths": seq_lengths,
            "forward_time": [],
            "memory_usage": [],
        }
        
        # Determine if model has embeddings (language modeling) or expects continuous input
        has_embeddings = self.model.token_emb is not None
        
        for seq_len in seq_lengths:
            # Create appropriate dummy input
            if has_embeddings:
                # Token indices for language modeling
                dummy_input = torch.randint(
                    0, 
                    self.model.vocab_size, 
                    (batch_size, seq_len),
                    device=self.device
                )
            else:
                # Continuous features
                dummy_input = torch.randn(
                    batch_size, 
                    seq_len, 
                    self.model.dim,
                    device=self.device
                )
            
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
    
    def get_eigenstate_activations(
        self,
        inputs: torch.Tensor,
        block_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract eigenstate activations from the model.
        
        Unlike traditional attention weights, TEN uses eigenstate dynamics.
        This method extracts the eigenstate magnitudes and phases for analysis.
        
        Args:
            inputs: Input tensor (batch, seq_len) or (batch, seq_len, dim)
            block_idx: Optional specific block index to extract from
            
        Returns:
            Dictionary with eigenstate information per block
        """
        eigenstate_info = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(module, type(self.model.blocks[0].cells[0])):
                    magnitude, phase = module.get_eigenvalues()
                    eigenstate_info[name] = {
                        'magnitude': magnitude.detach().cpu(),
                        'phase': phase.detach().cpu(),
                    }
            return hook
        
        # Register hooks
        hooks = []
        for i, block in enumerate(self.model.blocks):
            if block_idx is None or i == block_idx:
                for j, cell in enumerate(block.cells):
                    hook_name = f"block_{i}_cell_{j}"
                    hook = cell.register_forward_hook(hook_fn(hook_name))
                    hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(inputs.to(self.device))
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return eigenstate_info
    
    def evaluate_language_modeling(
        self,
        test_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate model on language modeling task.
        
        Computes perplexity and other language modeling metrics.
        Expects test_loader to yield sequences of token indices.
        
        Args:
            test_loader: DataLoader yielding token sequences (batch, seq_len)
            
        Returns:
            Dictionary with perplexity, loss, and token accuracy
        """
        total_loss = 0.0
        total_tokens = 0
        correct_tokens = 0
        criterion = nn.CrossEntropyLoss(reduction='sum')
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating LM"):
                if isinstance(batch, (list, tuple)):
                    tokens = batch[0]  # Handle (tokens,) or (tokens, labels) format
                else:
                    tokens = batch
                    
                tokens = tokens.to(self.device)
                batch_size, seq_len = tokens.shape
                
                # Prepare inputs and targets for language modeling
                inputs = tokens[:, :-1]  # All but last token
                targets = tokens[:, 1:]  # All but first token
                
                # Forward pass
                logits = self.model(inputs)
                
                # Reshape for loss computation
                logits_flat = logits.reshape(-1, logits.size(-1))
                targets_flat = targets.reshape(-1)
                
                # Compute loss
                loss = criterion(logits_flat, targets_flat)
                total_loss += loss.item()
                total_tokens += targets_flat.size(0)
                
                # Compute accuracy
                predictions = logits_flat.argmax(dim=-1)
                correct_tokens += (predictions == targets_flat).sum().item()
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        accuracy = correct_tokens / total_tokens
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "token_accuracy": accuracy,
        }


if __name__ == "__main__":
    from .model import TemporalEigenstateConfig
    
    # Example usage
    config = TemporalEigenstateConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_seq_len=2048,  # Increased to test longer sequences
        vocab_size=10000,
    )
    
    model = TemporalEigenstateNetwork(config)
    evaluator = Evaluator(model)
    
    print("Testing TEN Evaluator...")
    print(f"Model parameters: {model.count_parameters() / 1e6:.2f}M")
    
    # Measure efficiency - only test up to configured max_seq_len
    seq_lengths = [64, 128, 256, 512, 1024]
    print(f"\nMeasuring efficiency for sequence lengths: {seq_lengths}")
    efficiency = evaluator.measure_efficiency(seq_lengths, batch_size=8, num_runs=3)
    print("Efficiency measurement complete!")
    print(f"Sequence lengths: {efficiency['seq_lengths']}")
    print(f"Forward times (s): {[f'{t:.4f}' for t in efficiency['forward_time']]}")
    print(f"Memory usage (MB): {[f'{m:.1f}' for m in efficiency['memory_usage']]}")
    
    # Test eigenstate extraction
    print("\nTesting eigenstate extraction...")
    dummy_tokens = torch.randint(0, 10000, (2, 128))
    eigenstate_info = evaluator.get_eigenstate_activations(dummy_tokens, block_idx=0)
    print(f"Extracted eigenstate info from {len(eigenstate_info)} cells")
    
    print("\n✓ All evaluator tests passed!")
