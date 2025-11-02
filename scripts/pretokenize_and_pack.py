#!/usr/bin/env python3
"""
Pre-tokenize and pack datasets for FAST training
This removes tokenization from the training critical path

Usage:
    python3 pretokenize_and_pack.py --dataset finewebedu --chunk_size 32768
    python3 pretokenize_and_pack.py --dataset wikitext-103 --chunk_size 16384
"""

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import json

def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize and pack datasets")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["finewebedu", "wikitext-103", "tinystories", "openwebtext"],
                       help="Dataset to pre-tokenize")
    parser.add_argument("--chunk_size", type=int, default=32768,
                       help="Chunk size for packing (default: 32768)")
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                       help="Tokenizer to use (default: gpt2)")
    parser.add_argument("--num_proc", type=int, default=8,
                       help="Number of processes for parallel tokenization")
    parser.add_argument("--batch_size", type=int, default=2000,
                       help="Batch size for tokenization")
    parser.add_argument("--output_dir", type=str, default="/root/ten_workspace/tokenized",
                       help="Output directory for tokenized chunks")
    parser.add_argument("--max_chunks", type=int, default=0,
                       help="Maximum number of chunks to create (0=unlimited)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 PRE-TOKENIZATION AND PACKING")
    print("=" * 80)
    print(f"\nDataset: {args.dataset}")
    print(f"Chunk size: {args.chunk_size:,} tokens")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Parallel processes: {args.num_proc}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {args.output_dir}")
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
    if hf_config:
        ds = load_dataset(hf_name, hf_config, split="train")
    else:
        ds = load_dataset(hf_name, split="train")
    
    print(f"  ✓ Dataset loaded: {len(ds):,} samples")
    
    # Tokenize in parallel
    print(f"\n⚡ Tokenizing with {args.num_proc} parallel processes...")
    
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
    
    ds_tokenized = ds.map(
        tokenize_batch,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=ds.column_names,
        desc="Tokenizing"
    )
    
    print(f"  ✓ Tokenization complete!")
    
    # Concatenate all tokens
    print(f"\n🔗 Concatenating tokens...")
    all_ids = []
    for example in tqdm(ds_tokenized, desc="Concatenating"):
        all_ids.extend(example["input_ids"])
    
    total_tokens = len(all_ids)
    print(f"  ✓ Total tokens: {total_tokens:,}")
    print(f"  ✓ Will create ~{total_tokens // args.chunk_size:,} chunks")
    
    # Pack into fixed-length chunks
    print(f"\n📦 Packing into {args.chunk_size}-token chunks...")
    outdir = Path(args.output_dir) / args.dataset
    outdir.mkdir(parents=True, exist_ok=True)
    
    chunks_created = 0
    chunks_to_create = args.max_chunks if args.max_chunks > 0 else float('inf')
    
    for i in tqdm(range(0, len(all_ids), args.chunk_size), desc="Saving chunks"):
        if chunks_created >= chunks_to_create:
            break
            
        chunk = all_ids[i:i + args.chunk_size]
        
        # Only save full chunks (no partial chunks at end)
        if len(chunk) < args.chunk_size:
            print(f"\n  ⚠️  Skipping partial chunk at end ({len(chunk)} tokens)")
            continue
        
        chunk_tensor = torch.tensor(chunk, dtype=torch.long)
        chunk_path = outdir / f"chunk_{chunks_created:06d}.pt"
        torch.save(chunk_tensor, chunk_path)
        chunks_created += 1
    
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
