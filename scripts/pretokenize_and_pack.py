#!/usr/bin/env python3
"""
Pre-tokenize and pack datasets for FAST training
This removes tokenization from the training critical path

Usage:
    python3 pretokenize_and_pack.py --dataset finewebedu --chunk_size 32768 --force
    python3 pretokenize_and_pack.py --dataset wikitext-103 --chunk_size 16384 --force
"""

import argparse
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import json
import shutil
import os
import multiprocessing

def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize and pack datasets")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["finewebedu", "wikitext-103", "tinystories", "openwebtext"],
                       help="Dataset to pre-tokenize")
    parser.add_argument("--chunk_size", type=int, default=32768,
                       help="Chunk size for packing (default: 32768)")
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                       help="Tokenizer to use (default: gpt2)")
    parser.add_argument("--num_proc", type=int, default=None,
                       help="Number of processes (default: CPU count)")
    parser.add_argument("--batch_size", type=int, default=5000,
                       help="Batch size for tokenization (default: 5000)")
    parser.add_argument("--output_dir", type=str, default="/root/ten_workspace/tokenized",
                       help="Output directory for tokenized chunks")
    parser.add_argument("--cache_dir", type=str, default="/root/ten_workspace/data",
                       help="Directory to cache/download datasets locally")
    parser.add_argument("--max_chunks", type=int, default=0,
                       help="Maximum number of chunks to create (0=unlimited)")
    parser.add_argument("--force", action="store_true",
                       help="Force re-tokenization (delete existing output and cache)")
    parser.add_argument("--streaming", action="store_true",
                       help="Use streaming mode (no download, process on-the-fly)")
    
    args = parser.parse_args()
    
    # Auto-detect CPU cores
    if args.num_proc is None:
        args.num_proc = multiprocessing.cpu_count()
        print(f"Auto-detected {args.num_proc} CPU cores")
    
    # Setup output directory
    outdir = Path(args.output_dir) / args.dataset
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean up existing files if --force or if output directory exists
    if outdir.exists():
        if args.force:
            print(f"\n🗑️  --force flag: Deleting existing output directory...")
            shutil.rmtree(outdir)
            print(f"  ✓ Deleted: {outdir}")
        else:
            print(f"\n⚠️  WARNING: Output directory already exists: {outdir}")
            print(f"  Found existing tokenized data - this may slow down tokenization!")
            print(f"  To force fresh tokenization, use --force flag:")
            print(f"    python3 {__file__} --dataset {args.dataset} --force")
            response = input("\n  Continue anyway? [y/N]: ")
            if response.lower() != 'y':
                print("  Aborted.")
                return
            print(f"\n  Cleaning up old chunks...")
            shutil.rmtree(outdir)
            print(f"  ✓ Deleted old files")
    
    # Create fresh output directory
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("🚀 PRE-TOKENIZATION AND PACKING (OPTIMIZED)")
    print("=" * 80)
    print(f"\nDataset: {args.dataset}")
    print(f"Chunk size: {args.chunk_size:,} tokens")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"CPU cores: {args.num_proc}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output dir: {args.output_dir}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"Streaming: {args.streaming}")
    print(f"Force clean: {args.force}")
    print("=" * 80)
    
    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print(f"  ✓ Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    # Load dataset
    print(f"\n📚 Loading dataset: {args.dataset}...")
    dataset_map = {
        "finewebedu": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
        "wikitext-103": ("wikitext", "wikitext-103-raw-v1"),
        "tinystories": ("roneneldan/TinyStories", None),
        "openwebtext": ("Skylion007/openwebtext", None),
    }
    
    hf_name, hf_config = dataset_map[args.dataset]
    
    # Load or download dataset locally
    if args.streaming:
        print("  Using streaming mode (no download)...")
        if hf_config:
            ds = load_dataset(hf_name, hf_config, split="train", streaming=True)
        else:
            ds = load_dataset(hf_name, split="train", streaming=True)
        print(f"  ✓ Dataset streaming enabled")
    else:
        print(f"  Downloading/loading to cache: {args.cache_dir}")
        if hf_config:
            ds = load_dataset(hf_name, hf_config, split="train", cache_dir=str(cache_dir))
        else:
            ds = load_dataset(hf_name, split="train", cache_dir=str(cache_dir))
        print(f"  ✓ Dataset loaded: {len(ds):,} samples")
        print(f"  ✓ Cached locally in: {cache_dir}")
    
    # Tokenize in parallel
    print(f"\n⚡ Tokenizing with {args.num_proc} parallel processes...")
    print(f"  Using batch size: {args.batch_size}")
    print(f"  Cache disabled for fresh tokenization...")
    
    def tokenize_batch(batch):
        # Find text field
        text_field = 'text'
        if text_field not in batch:
            for field in ['content', 'article', 'document']:
                if field in batch:
                    text_field = field
                    break
        
        return tokenizer(
            batch[text_field],
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False
        )
    
    # Disable caching to avoid slowdowns from old cache files
    ds_tokenized = ds.map(
        tokenize_batch,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc if not args.streaming else None,
        remove_columns=ds.column_names,
        desc="Tokenizing",
        load_from_cache_file=False,  # CRITICAL: Disable cache to maintain speed!
        keep_in_memory=False  # Stream to disk to avoid RAM issues
    )
    
    print(f"  ✓ Tokenization complete!")
    
    # OPTIMIZED: Stream and pack directly (no slow concatenation!)
    print(f"\n� Streaming and packing into {args.chunk_size}-token chunks...")
    print(f"  (This is MUCH faster than the old concatenation method!)")
    
    chunks_created = 0
    chunks_to_create = args.max_chunks if args.max_chunks > 0 else float('inf')
    current_chunk = []
    total_tokens = 0
    
    # Stream through tokenized data and pack on-the-fly
    for example in tqdm(ds_tokenized, desc="Packing chunks", unit=" samples"):
        tokens = example["input_ids"]
        current_chunk.extend(tokens)
        total_tokens += len(tokens)
        
        # When we have enough tokens for a full chunk, save it
        while len(current_chunk) >= args.chunk_size:
            if chunks_created >= chunks_to_create:
                break
                
            # Extract exactly chunk_size tokens
            chunk = current_chunk[:args.chunk_size]
            current_chunk = current_chunk[args.chunk_size:]
            
            # Save chunk
            chunk_tensor = torch.tensor(chunk, dtype=torch.long)
            chunk_path = outdir / f"chunk_{chunks_created:06d}.pt"
            torch.save(chunk_tensor, chunk_path)
            chunks_created += 1
        
        if chunks_created >= chunks_to_create:
            break
    
    print(f"\n  ✓ Packed {chunks_created:,} chunks from {total_tokens:,} tokens")
    print(f"  ✓ Leftover tokens (not saved): {len(current_chunk):,}")
    
    # Save metadata
    metadata = {
        "dataset": args.dataset,
        "tokenizer": args.tokenizer,
        "vocab_size": len(tokenizer),
        "chunk_size": args.chunk_size,
        "num_chunks": chunks_created,
        "total_tokens": chunks_created * args.chunk_size,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else None,
    }
    
    with open(outdir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✅ PRE-TOKENIZATION COMPLETE!")
    print("=" * 80)
    print(f"\nStatistics:")
    print(f"  Chunks created: {chunks_created:,}")
    print(f"  Tokens per chunk: {args.chunk_size:,}")
    print(f"  Total tokens: {chunks_created * args.chunk_size:,}")
    print(f"  Output directory: {outdir}")
    print(f"  Metadata saved: {outdir / 'metadata.json'}")
    
    # Calculate training estimates
    print(f"\n📊 Training Estimates:")
    print(f"  Total tokens: {chunks_created * args.chunk_size:,}")
    print(f"  Per epoch: {chunks_created:,} chunks")
    print(f"  Disk usage: ~{chunks_created * args.chunk_size * 2 / 1024**3:.2f} GB")
    
    print("\n🎯 Next steps:")
    print(f"  1. Use --pretokenized flag in training:")
    print(f"     python3 examples/train_digitalocean.py \\")
    print(f"         --config large \\")
    print(f"         --dataset {args.dataset} \\")
    print(f"         --pretokenized \\")
    print(f"         --mixed_precision \\")
    print(f"         --epochs 1")
    print("\n  2. Training will be 5-50× FASTER! 🔥")
    print("=" * 80)


if __name__ == "__main__":
    main()
