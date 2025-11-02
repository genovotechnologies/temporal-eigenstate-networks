"""
Example: Multi-Modal TEN for Vision + Text

This example demonstrates how to use TEN_MultiModal for tasks combining
multiple modalities like image captioning, visual question answering,
video understanding with text, etc.

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


class MultiModalDataset(Dataset):
    """Dummy multi-modal dataset for demonstration."""
    def __init__(
        self, 
        num_samples: int, 
        text_seq_len: int, 
        vocab_size: int,
        image_features: int,
        num_classes: int
    ):
        self.num_samples = num_samples
        self.text_seq_len = text_seq_len
        self.vocab_size = vocab_size
        self.image_features = image_features
        self.num_classes = num_classes
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Text input (e.g., question or caption)
        text = torch.randint(0, self.vocab_size, (self.text_seq_len,))
        
        # Image features (e.g., from CNN like ResNet)
        # In practice: extract features from pre-trained vision model
        # Shape: (num_patches, feature_dim) e.g., (49, 2048) for 7x7 patches
        image = torch.randn(49, self.image_features)
        
        # Label (for classification task)
        label = torch.randint(0, self.num_classes, (1,)).item()
        
        return {'text': text, 'image': image}, label


def main():
    print("=" * 80)
    print("Multi-Modal TEN: Vision + Text Example")
    print("=" * 80)
    
    # Configuration
    vocab_size = 10000
    text_seq_len = 64
    image_features = 2048  # e.g., ResNet features
    num_classes = 100  # e.g., 100 different categories
    
    # Create Multi-Modal TEN
    print("\nCreating Multi-Modal TEN...")
    
    modality_configs = {
        'text': {
            'input_dim': vocab_size,
            'd_model': 512,
            'input_type': 'discrete',  # Token indices
        },
        'image': {
            'input_dim': image_features,
            'd_model': 512,
            'input_type': 'continuous',  # Feature vectors
        }
    }
    
    model = create_model(
        model_type='multimodal',
        modality_configs=modality_configs,
        fusion_dim=512,
        num_fusion_layers=2,
        num_eigenstates=64,
        num_classes=num_classes,
        task_type='classification',
        max_seq_len=max(text_seq_len, 49),  # Max of text and image patches
        dropout=0.1,
    )
    
    print(f"✓ Model created with {model.count_parameters() / 1e6:.2f}M parameters")
    print(f"  Modalities: {model.modality_names}")
    print(f"  Fusion dimension: {model.fusion_dim}")
    print(f"  Task: {model.task_type}")
    
    # Create datasets
    print("\nPreparing datasets...")
    train_dataset = MultiModalDataset(1000, text_seq_len, vocab_size, image_features, num_classes)
    val_dataset = MultiModalDataset(200, text_seq_len, vocab_size, image_features, num_classes)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Setup training
    print("\nSetting up training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        language_modeling=False,
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
    
    test_text = torch.randint(0, vocab_size, (1, text_seq_len))
    test_image = torch.randn(1, 49, image_features)
    test_inputs = {'text': test_text, 'image': test_image}
    
    with torch.no_grad():
        # Classification output
        logits = model(test_inputs)
        probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        
        # Get embeddings for retrieval
        embeddings = model(test_inputs, return_embeddings=True)
    
    print(f"  Text input shape: {test_text.shape}")
    print(f"  Image input shape: {test_image.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Predicted class: {predicted_class}")
    print(f"  Confidence: {probs[0, predicted_class]:.4f}")
    print(f"  Embedding shape: {embeddings.shape}")
    
    # Test with single modality
    print("\nTesting with single modality (text only)...")
    with torch.no_grad():
        text_only_logits = model({'text': test_text})
    print(f"  Text-only output: {text_only_logits.shape}")
    
    print("\n✓ Example complete!")
    
    print("\n" + "=" * 80)
    print("Use Cases for Multi-Modal TEN:")
    print("=" * 80)
    print("1. Image Captioning: image → text generation")
    print("2. Visual Question Answering: image + question → answer")
    print("3. Video Understanding: video frames + audio → action recognition")
    print("4. Document Understanding: text + layout features → classification")
    print("5. Cross-Modal Retrieval: find images from text or vice versa")
    print("6. Multi-Sensor Fusion: combine sensor data for robust predictions")
    print("=" * 80)


if __name__ == "__main__":
    main()
