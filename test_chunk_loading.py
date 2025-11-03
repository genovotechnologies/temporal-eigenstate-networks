#!/usr/bin/env python3
"""Test if chunk loading works"""

import torch
from pathlib import Path
import json
import time

print("Testing chunk loading...")

tokenized_dir = Path("/root/ten_workspace/tokenized/finewebedu")

# Check if directory exists
print(f"\n1. Directory exists: {tokenized_dir.exists()}")

if tokenized_dir.exists():
    # Check metadata
    metadata_path = tokenized_dir / "metadata.json"
    print(f"2. Metadata exists: {metadata_path.exists()}")
    
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"3. Metadata content: {metadata}")
    
    # Check chunk files
    chunk_files = list(tokenized_dir.glob("chunk_*.pt"))
    print(f"4. Found {len(chunk_files)} chunk files")
    
    if chunk_files:
        # Try loading first chunk
        print(f"\n5. Testing load of first chunk: {chunk_files[0]}")
        start = time.time()
        
        try:
            chunk = torch.load(chunk_files[0], weights_only=True)
            elapsed = time.time() - start
            print(f"   ✓ Loaded successfully in {elapsed:.3f}s")
            print(f"   Shape: {chunk.shape}")
            print(f"   Dtype: {chunk.dtype}")
            print(f"   Size: {chunk.numel() * chunk.element_size() / 1024 / 1024:.1f} MB")
        except Exception as e:
            print(f"   ✗ Failed to load: {e}")
        
        # Try loading 10 chunks to see if it's consistently slow
        print(f"\n6. Loading 10 random chunks...")
        import random
        test_chunks = random.sample(chunk_files[:1000], min(10, len(chunk_files)))
        
        for i, chunk_file in enumerate(test_chunks):
            start = time.time()
            try:
                chunk = torch.load(chunk_file, weights_only=True)
                elapsed = time.time() - start
                print(f"   Chunk {i+1}: {elapsed:.3f}s - shape {chunk.shape}")
            except Exception as e:
                print(f"   Chunk {i+1}: FAILED - {e}")
                break
else:
    print("ERROR: Tokenized directory not found!")
    print("Expected location: /root/ten_workspace/tokenized/finewebedu")
