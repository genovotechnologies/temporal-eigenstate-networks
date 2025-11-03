#!/usr/bin/env python3
"""Minimal test to isolate DataLoader hang"""

import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import json
import time


class SimplePreTokenizedDataset(Dataset):
    """Minimal dataset without caching"""
    
    def __init__(self, chunks_dir):
        self.chunks_dir = Path(chunks_dir)
        self.chunk_files = sorted(self.chunks_dir.glob("chunk_*.pt"))
        
        if not self.chunk_files:
            raise ValueError(f"No chunks found in {chunks_dir}")
        
        print(f"Found {len(self.chunk_files)} chunks")
    
    def __len__(self):
        return len(self.chunk_files)
    
    def __getitem__(self, idx):
        print(f"  [Worker] Loading chunk {idx}...")
        chunk = torch.load(self.chunk_files[idx], weights_only=True)
        print(f"  [Worker] Loaded chunk {idx}: shape {chunk.shape}")
        return chunk


def run_tests():
    """Run all DataLoader tests"""
    print("=" * 80)
    print("DataLoader Hang Test")
    print("=" * 80)

    tokenized_dir = "/root/ten_workspace/tokenized/finewebedu"

    print(f"\n1. Creating dataset...")
    dataset = SimplePreTokenizedDataset(tokenized_dir)

    print(f"\n2. Testing single item access (no DataLoader)...")
    start = time.time()
    item = dataset[0]
    print(f"   ✓ Got item in {time.time() - start:.2f}s: shape {item.shape}")

    print(f"\n3. Testing DataLoader with num_workers=0 (single process)...")
    loader = DataLoader(dataset, batch_size=2, num_workers=0)
    start = time.time()
    for i, batch in enumerate(loader):
        print(f"   Batch {i+1}: {batch.shape} (elapsed: {time.time() - start:.1f}s)")
        if i >= 2:  # Only test first 3 batches
            break
    print(f"   ✓ Single-process DataLoader works!")

    print(f"\n4. Testing DataLoader with num_workers=2, spawn, NO persistence...")
    loader = DataLoader(
        dataset, 
        batch_size=2, 
        num_workers=2,
        multiprocessing_context='spawn',
        persistent_workers=False,  # Disable persistence
        prefetch_factor=2
    )
    print("   Starting iteration (may take 5-10s for worker spawn)...")
    start = time.time()

    try:
        for i, batch in enumerate(loader):
            elapsed = time.time() - start
            print(f"   Batch {i+1}: {batch.shape} (elapsed: {elapsed:.1f}s)")
            if i >= 2:
                break
        print(f"   ✓ Multi-process DataLoader works!")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")

    print(f"\n5. Testing with persistent_workers=True...")
    loader = DataLoader(
        dataset, 
        batch_size=2, 
        num_workers=2,
        multiprocessing_context='spawn',
        persistent_workers=True,  # Enable persistence
        prefetch_factor=2
    )
    print("   Starting iteration (checking if persistent workers cause hang)...")
    start = time.time()

    # Set a timeout
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("DataLoader iteration timeout!")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30 second timeout

    try:
        for i, batch in enumerate(loader):
            elapsed = time.time() - start
            print(f"   Batch {i+1}: {batch.shape} (elapsed: {elapsed:.1f}s)")
            if i >= 2:
                break
        signal.alarm(0)  # Cancel timeout
        print(f"   ✓ Persistent workers work!")
    except TimeoutError:
        print(f"   ✗ TIMEOUT! Persistent workers cause hang!")
    except Exception as e:
        print(f"   ✗ FAILED: {e}")
    finally:
        signal.alarm(0)

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == '__main__':
    run_tests()

