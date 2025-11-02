"""
Multi-Modal Utilities for Temporal Eigenstate Networks
========================================================

Copyright (c) 2025 Genovo Technologies. All Rights Reserved.
PROPRIETARY AND CONFIDENTIAL - Internal Use Only

This module provides utilities for working with multi-modal TEN models,
including data preprocessing, feature extraction, and evaluation helpers.

Author: Oluwatosin Afolabi
Company: Genovo Technologies
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class MultiModalPreprocessor:
    """
    Preprocessor for multi-modal data.
    
    Handles common preprocessing tasks like:
    - Text tokenization and padding
    - Image feature extraction
    - Audio feature extraction
    - Normalization and standardization
    """
    
    def __init__(
        self,
        modalities: List[str],
        text_vocab_size: Optional[int] = None,
        max_seq_len: int = 512,
    ):
        """
        Initialize preprocessor.
        
        Args:
            modalities: List of modality names (e.g., ['text', 'image', 'audio'])
            text_vocab_size: Vocabulary size for text tokenization
            max_seq_len: Maximum sequence length for each modality
        """
        self.modalities = modalities
        self.text_vocab_size = text_vocab_size
        self.max_seq_len = max_seq_len
        
    def preprocess_text(
        self, 
        tokens: Union[List[int], torch.Tensor],
        pad_value: int = 0
    ) -> torch.Tensor:
        """
        Preprocess text tokens.
        
        Args:
            tokens: List or tensor of token indices
            pad_value: Value to use for padding
            
        Returns:
            Padded tensor of shape (seq_len,)
        """
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, dtype=torch.long)
        
        # Truncate if too long
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        
        # Pad if too short
        if len(tokens) < self.max_seq_len:
            padding = torch.full((self.max_seq_len - len(tokens),), pad_value, dtype=torch.long)
            tokens = torch.cat([tokens, padding])
        
        return tokens
    
    def preprocess_image_features(
        self,
        features: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Preprocess image features (e.g., from CNN).
        
        Args:
            features: Image features, shape (num_patches, feature_dim)
            
        Returns:
            Normalized features tensor
        """
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        
        # Normalize features
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        
        return features
    
    def preprocess_audio_features(
        self,
        features: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Preprocess audio features (e.g., mel spectrograms).
        
        Args:
            features: Audio features, shape (time_steps, feature_dim)
            
        Returns:
            Normalized features tensor
        """
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        
        # Standardize features
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True) + 1e-8
        features = (features - mean) / std
        
        return features
    
    def batch_preprocess(
        self,
        batch: Dict[str, List]
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess a batch of multi-modal data.
        
        Args:
            batch: Dictionary mapping modality names to lists of samples
            
        Returns:
            Dictionary of preprocessed tensors
        """
        processed = {}
        
        for modality in self.modalities:
            if modality not in batch:
                continue
                
            samples = batch[modality]
            
            if modality == 'text':
                processed[modality] = torch.stack([
                    self.preprocess_text(s) for s in samples
                ])
            elif modality == 'image':
                processed[modality] = torch.stack([
                    self.preprocess_image_features(s) for s in samples
                ])
            elif modality == 'audio':
                processed[modality] = torch.stack([
                    self.preprocess_audio_features(s) for s in samples
                ])
            else:
                # Default: just convert to tensor
                if isinstance(samples[0], torch.Tensor):
                    processed[modality] = torch.stack(samples)
                else:
                    processed[modality] = torch.tensor(samples, dtype=torch.float32)
        
        return processed


class ModalityFusion:
    """
    Utilities for fusing representations from multiple modalities.
    """
    
    @staticmethod
    def concatenate_fusion(
        embeddings: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Simple concatenation fusion.
        
        Args:
            embeddings: Dict of modality embeddings, each (batch, dim)
            
        Returns:
            Concatenated tensor (batch, sum_of_dims)
        """
        return torch.cat(list(embeddings.values()), dim=-1)
    
    @staticmethod
    def weighted_sum_fusion(
        embeddings: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Weighted sum fusion.
        
        Args:
            embeddings: Dict of modality embeddings, each (batch, dim)
            weights: Optional weights for each modality
            
        Returns:
            Fused tensor (batch, dim)
        """
        if weights is None:
            weights = {k: 1.0 for k in embeddings.keys()}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        fused = None
        for modality, emb in embeddings.items():
            weighted = emb * weights[modality]
            fused = weighted if fused is None else fused + weighted
        
        return fused
    
    @staticmethod
    def attention_fusion(
        query_emb: torch.Tensor,
        key_value_embs: Dict[str, torch.Tensor],
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Attention-based fusion where one modality queries others.
        
        Args:
            query_emb: Query embedding (batch, dim)
            key_value_embs: Dict of key/value embeddings
            temperature: Softmax temperature
            
        Returns:
            Fused tensor (batch, dim)
        """
        # Compute attention scores
        scores = {}
        for modality, kv_emb in key_value_embs.items():
            # Cosine similarity
            score = torch.sum(query_emb * kv_emb, dim=-1) / temperature
            scores[modality] = score
        
        # Softmax over modalities
        score_tensor = torch.stack(list(scores.values()), dim=1)  # (batch, num_modalities)
        attn_weights = torch.softmax(score_tensor, dim=1)
        
        # Weighted combination
        kv_tensor = torch.stack(list(key_value_embs.values()), dim=1)  # (batch, num_mod, dim)
        fused = torch.sum(attn_weights.unsqueeze(-1) * kv_tensor, dim=1)
        
        return fused


class MultiModalMetrics:
    """
    Evaluation metrics for multi-modal models.
    """
    
    @staticmethod
    def cross_modal_retrieval_recall(
        query_embs: torch.Tensor,
        key_embs: torch.Tensor,
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """
        Compute recall@k for cross-modal retrieval.
        
        Args:
            query_embs: Query embeddings (N, dim)
            key_embs: Key embeddings (N, dim)
            k_values: List of k values for recall@k
            
        Returns:
            Dictionary of recall@k scores
        """
        # Compute similarity matrix
        similarity = torch.matmul(query_embs, key_embs.t())  # (N, N)
        
        # Get rankings
        _, rankings = torch.sort(similarity, dim=1, descending=True)
        
        # Compute recall@k
        results = {}
        N = query_embs.size(0)
        
        for k in k_values:
            # Count how many times the correct item is in top-k
            correct = 0
            for i in range(N):
                if i in rankings[i, :k]:
                    correct += 1
            
            results[f'recall@{k}'] = correct / N
        
        return results
    
    @staticmethod
    def compute_alignment_score(
        emb1: torch.Tensor,
        emb2: torch.Tensor
    ) -> float:
        """
        Compute alignment score between two modality embeddings.
        
        Uses cosine similarity averaged across the batch.
        
        Args:
            emb1: Embeddings from modality 1 (batch, dim)
            emb2: Embeddings from modality 2 (batch, dim)
            
        Returns:
            Mean alignment score
        """
        # Normalize
        emb1 = emb1 / (emb1.norm(dim=-1, keepdim=True) + 1e-8)
        emb2 = emb2 / (emb2.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Cosine similarity
        similarity = torch.sum(emb1 * emb2, dim=-1)
        
        return similarity.mean().item()


def extract_visual_features(
    images: torch.Tensor,
    vision_model: nn.Module,
    pool_method: str = 'spatial'
) -> torch.Tensor:
    """
    Extract visual features from images using a pre-trained model.
    
    Args:
        images: Input images (batch, channels, height, width)
        vision_model: Pre-trained vision model (e.g., ResNet)
        pool_method: Pooling method ('spatial', 'global_avg', 'global_max')
        
    Returns:
        Visual features (batch, num_patches, feature_dim) or (batch, feature_dim)
    """
    with torch.no_grad():
        features = vision_model(images)
    
    if pool_method == 'global_avg':
        features = features.mean(dim=[2, 3])  # (batch, channels)
    elif pool_method == 'global_max':
        features = features.amax(dim=[2, 3])  # (batch, channels)
    elif pool_method == 'spatial':
        # Keep spatial dimensions as patches
        batch, channels, h, w = features.shape
        features = features.view(batch, channels, h * w).permute(0, 2, 1)  # (batch, h*w, channels)
    
    return features


def create_multimodal_collate_fn(
    modalities: List[str],
    pad_values: Optional[Dict[str, int]] = None
):
    """
    Create a collate function for multi-modal data.
    
    Args:
        modalities: List of modality names
        pad_values: Dictionary of padding values for each modality
        
    Returns:
        Collate function for DataLoader
    """
    if pad_values is None:
        pad_values = {m: 0 for m in modalities}
    
    def collate_fn(batch):
        """Collate multi-modal samples into a batch."""
        collated = {modality: [] for modality in modalities}
        labels = []
        
        for sample in batch:
            data, label = sample
            for modality in modalities:
                if modality in data:
                    collated[modality].append(data[modality])
            labels.append(label)
        
        # Stack tensors
        for modality in modalities:
            if collated[modality]:
                collated[modality] = torch.stack(collated[modality])
        
        labels = torch.tensor(labels, dtype=torch.long)
        
        return collated, labels
    
    return collate_fn


__all__ = [
    'MultiModalPreprocessor',
    'ModalityFusion',
    'MultiModalMetrics',
    'extract_visual_features',
    'create_multimodal_collate_fn',
]
