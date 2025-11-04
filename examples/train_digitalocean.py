"""
Optimized Training Script for DigitalOcean L40S/RTX 6000 Ada (48GB)
Updated for Paper-Compliant TEN Implementation

This script uses the new paper-compliant features:
- Proper eigenvalue initialization (Appendix B.2)
- Hierarchical TEN (HTEN) support (Section 5)
- Energy regularization (Theorem 4)
- Chunk-based processing with gradient checkpointing
- Efficient positional embeddings
- Mixed precision (FP16) training

Optimized for:
- Long-range tasks (8192 token sequences)
- Large model training (1024 dim, 8+ layers)
- 48GB VRAM efficiency
- 5-hour training window

Usage:
    python train_digitalocean.py --config large
    python train_digitalocean.py --config medium --max_seq_len 4096
    python train_digitalocean.py --config large --use_hten  # Enable Hierarchical TEN
    python train_digitalocean.py --benchmark
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
import time
import os
import sys
import gc
from pathlib import Path
from tqdm import tqdm
import json
import multiprocessing as mp

# CRITICAL: Set start method for multiprocessing BEFORE any CUDA operations
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

# Enable cuDNN benchmarking for faster training
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Enable TF32 for 2-3× speedup on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import (
    TemporalEigenstateConfig, 
    TemporalEigenstateNetwork,
    print_model_summary,
    count_parameters
)


# Fast pre-tokenized dataset loader
class PreTokenizedDataset(Dataset):
    """Fast dataset loader for pre-tokenized and packed chunks
    
    Loads chunks from disk on-demand with LRU caching.
    Uses memory-mapped loading for speed.
    Automatically truncates chunks to max_seq_len if needed.
    """
    
    def __init__(self, chunks_dir, cache_size=128, max_seq_len=None):
        self.chunks_dir = Path(chunks_dir)
        self.chunk_files = sorted(self.chunks_dir.glob("chunk_*.pt"))
        self.cache_size = cache_size
        self.max_seq_len = max_seq_len
        self._cache = {}
        self._cache_order = []
        
        if not self.chunk_files:
            raise ValueError(f"No chunks found in {chunks_dir}")
        
        # Load metadata
        with open(self.chunks_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        
        print(f"  Found {len(self.chunk_files)} pre-tokenized chunks")
        print(f"  Chunk size: {self.metadata['chunk_size']:,} tokens")
        
        # Warn if truncation will occur
        if max_seq_len and max_seq_len < self.metadata['chunk_size']:
            print(f"  ⚠️  Chunks will be TRUNCATED from {self.metadata['chunk_size']:,} to {max_seq_len:,} tokens")
            print(f"  This wastes {(1 - max_seq_len/self.metadata['chunk_size'])*100:.1f}% of pre-tokenized data")
            effective_tokens = len(self.chunk_files) * max_seq_len
        else:
            effective_tokens = self.metadata['total_tokens']
        
        print(f"  Effective tokens: {effective_tokens:,}")
        
        # Calculate total disk usage
        total_size_gb = len(self.chunk_files) * self.metadata['chunk_size'] * 2 / 1024**3
        print(f"  Total size: ~{total_size_gb:.1f}GB on disk")
        print(f"  ✓ LRU cache: {cache_size} chunks in RAM (~{cache_size * self.metadata['chunk_size'] * 2 / 1024**2:.0f}MB)")
        print(f"  ✓ Chunks loaded on-demand with caching!")
    
    def __len__(self):
        return len(self.chunk_files)
    
    def __getitem__(self, idx):
        # Check cache first
        if idx in self._cache:
            chunk = self._cache[idx]
        else:
            # Load chunk from disk
            chunk = torch.load(self.chunk_files[idx], weights_only=True)
            
            # Add to cache with LRU eviction
            if len(self._cache) >= self.cache_size:
                # Remove oldest item
                oldest_idx = self._cache_order.pop(0)
                del self._cache[oldest_idx]
            
            self._cache[idx] = chunk
            self._cache_order.append(idx)
        
        # Truncate if needed
        if self.max_seq_len and chunk.size(0) > self.max_seq_len:
            chunk = chunk[:self.max_seq_len]
        
        return chunk


# Predefined configurations optimized for 48GB GPU
CONFIGS = {
    "nano": {
        "d_model": 512,
        "n_layers": 4,
        "num_eigenstates": 64,
        "batch_size": 8,
        "max_seq_len": 512,
        "use_hten": False,
        "energy_reg_weight": 0.01,
        "description": "Nano - 33M params (~0.6GB model + 2GB activations = 2.6GB total)",
    },
    "micro": {
        "d_model": 768,
        "n_layers": 6,
        "num_eigenstates": 96,
        "batch_size": 16,
        "max_seq_len": 1024,
        "use_hten": False,
        "energy_reg_weight": 0.01,
        "description": "Micro - 85M params (~1.5GB model + 4GB activations = 5.5GB total)",
    },
    "small": {
        "d_model": 1024,
        "n_layers": 8,
        "num_eigenstates": 128,
        "batch_size": 16,
        "max_seq_len": 1024,
        "use_hten": False,
        "energy_reg_weight": 0.01,
        "description": "Small - 180M params (~2.5GB model + 5GB activations = 7.5GB total)",
    },
    "medium": {
        "d_model": 1280,
        "n_layers": 12,
        "num_eigenstates": 160,
        "batch_size": 16,
        "max_seq_len": 1024,
        "use_hten": False,
        "energy_reg_weight": 0.01,
        "description": "Medium - 320M params (~4GB model + 6GB activations = 10GB total)",
    },
    "large": {
        "d_model": 1536,
        "n_layers": 16,
        "num_eigenstates": 192,
        "batch_size": 8,
        "max_seq_len": 1024,
        "use_hten": True,  # Enable HTEN for large model
        "hten_scales": [1, 2, 4],
        "energy_reg_weight": 0.02,  # Higher weight for stability
        "description": "Large - 520M params (~6GB model + 8GB activations = 14GB total)",
    },
    "xlarge": {
        "d_model": 2048,
        "n_layers": 20,
        "num_eigenstates": 256,
        "batch_size": 8,
        "max_seq_len": 2048,
        "use_hten": True,  # Enable HTEN for xlarge model
        "hten_scales": [1, 2, 4, 8],
        "energy_reg_weight": 0.02,  # Higher weight for stability
        "description": "XLarge - 1.2B params (~12GB model + 16GB activations = 28GB total)",
    },
}


class DigitalOceanTrainer:
    """Optimized trainer for DigitalOcean L40S/RTX 6000 Ada GPUs"""
    
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.device = torch.device('cuda')
        self.start_time = time.time()
        self.hourly_rate = 1.57  # DigitalOcean rate
        
        # Setup output directories
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        print("=" * 80)
        print(f"DigitalOcean GPU Training - {args.config.upper()} Configuration")
        print("=" * 80)
        
    def print_cost(self):
        """Print current cost"""
        elapsed = time.time() - self.start_time
        hours = elapsed / 3600
        cost = hours * self.hourly_rate
        remaining = 15.00 - cost  # $15 free credit
        
        print(f"\n⏱️  Elapsed: {elapsed/60:.1f} min | Cost: ${cost:.2f} | Remaining: ${remaining:.2f}")
        
    def create_model(self):
        """Create and configure model with new paper-compliant features"""
        print(f"\n📦 Creating model with paper-compliant TEN implementation...")
        
        # Create configuration with new features
        model_config = TemporalEigenstateConfig(
            vocab_size=self.args.vocab_size,
            dim=self.config['d_model'],
            n_layers=self.config['n_layers'],
            num_eigenstates=self.config['num_eigenstates'],
            num_cells=self.config.get('num_cells', 2),
            max_seq_len=self.config['max_seq_len'],
            dropout=self.config.get('dropout', 0.1),
            
            # Memory optimizations
            chunk_size=self.config.get('chunk_size', 64),
            use_gradient_checkpointing=self.config.get('use_gradient_checkpointing', True),
            
            # Paper-compliant features
            use_resonance=self.config.get('use_resonance', True),
            resonance_epsilon=self.config.get('resonance_epsilon', 0.01),
            eigenvalue_clip=self.config.get('eigenvalue_clip', 0.99),
            ffn_multiplier=self.config.get('ffn_multiplier', 4.0),
            
            # Positional embeddings (learned is more efficient for training)
            pos_emb_type=self.config.get('pos_emb_type', 'learned'),
            
            # Energy regularization (Theorem 4)
            energy_reg_weight=self.config.get('energy_reg_weight', 0.01),
            
            # Hierarchical TEN (Section 5) - optional
            use_hten=self.config.get('use_hten', False),
            hten_scales=self.config.get('hten_scales', [1, 2, 4, 8]),
            
            tie_weights=True,  # Save memory
        )
        
        model = TemporalEigenstateNetwork(model_config).to(self.device)
        
        # Print detailed model summary
        print_model_summary(model, verbose=True)
        
        # Additional training info
        num_params = count_parameters(model)
        print(f"\n💾 Memory Estimates:")
        print(f"  Model (FP32): ~{num_params * 4 / 1024**3:.2f} GB")
        print(f"  Model (FP16): ~{num_params * 2 / 1024**3:.2f} GB")
        print(f"  Training (FP16 + optimizer): ~{num_params * 10 / 1024**3:.2f} GB")
        
        # Test forward pass
        print(f"\n🧪 Testing forward pass...")
        test_input = torch.randint(0, self.args.vocab_size, (2, 128)).to(self.device)
        with torch.no_grad():
            test_output = model(test_input)
        print(f"  ✓ Forward pass successful: {test_input.shape} -> {test_output.shape}")
        
        # Test loss computation with energy regularization
        test_targets = torch.randint(0, self.args.vocab_size, (2, 128)).to(self.device)
        with torch.no_grad():
            loss_dict = model.compute_loss(test_input, test_targets, return_dict=True)
            print(f"  ✓ Loss computation successful:")
            print(f"    CE Loss: {loss_dict['ce_loss'].item():.4f}")
            print(f"    Energy Loss: {loss_dict['energy_loss'].item():.4f}")
            print(f"    Total Loss: {loss_dict['loss'].item():.4f}")
        
        return model
    
    def prepare_data(self):
        """Load and prepare dataset"""
        print(f"\n📚 Loading dataset: {self.args.dataset}...")

        # Map friendly dataset names to HuggingFace identifiers
        dataset_map = {
            "imdb": ("imdb", None),
            "ag_news": ("ag_news", None),
            "wikitext-103": ("wikitext", "wikitext-103-raw-v1"),
            "tinystories": ("roneneldan/TinyStories", None),
            "finewebedu": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
            "pg19": ("emozilla/pg19-test", None),
            "openwebtext": ("Skylion007/openwebtext", None),
        }

        if self.args.dataset not in dataset_map:
            raise ValueError(f"Unknown dataset: {self.args.dataset}")

        hf_name, hf_config = dataset_map[self.args.dataset]
        
        # Use streaming for large datasets to avoid full download
        use_streaming = self.args.streaming or self.args.dataset in ["finewebedu", "openwebtext", "pg19"]
        
        if use_streaming:
            print(f"  🌊 Using streaming mode (no full download needed)...")
            # Load dataset in streaming mode
            if hf_config:
                dataset = load_dataset(hf_name, hf_config, split="train", streaming=True)
            else:
                dataset = load_dataset(hf_name, split="train", streaming=True)
            
            print(f"  ✓ Dataset streaming enabled")
            print(f"  ⚡ Training will start immediately (no waiting for full download!)")
            train_len = "streaming"
            test_len = 0
            num_classes = None  # language modeling / unsupervised
            
            # Wrap in dict for compatibility
            dataset = {'train': dataset}
            
        else:
            # Load dataset normally (full download)
            if hf_config:
                dataset = load_dataset(hf_name, hf_config)
            else:
                dataset = load_dataset(hf_name)

            # Determine task type
            if self.args.dataset in ("imdb", "ag_news"):
                num_classes = 2 if self.args.dataset == "imdb" else 4
            else:
                num_classes = None  # language modeling / unsupervised

            print(f"  ✓ Dataset loaded")
            # Some datasets don't have a test split
            train_len = len(dataset['train']) if 'train' in dataset else 0
            test_len = len(dataset['test']) if 'test' in dataset else 0
        
        print(f"  Train samples: {train_len if isinstance(train_len, str) else f'{train_len:,}'}")
        if test_len > 0:
            print(f"  Test samples: {test_len:,}")

        # Use subset if specified (only for non-streaming)
        if self.args.subset_size > 0 and not use_streaming:
            if isinstance(train_len, int) and train_len > 0:
                sel = min(self.args.subset_size, train_len)
                dataset['train'] = dataset['train'].shuffle(seed=42).select(range(sel))
                if test_len > 0:
                    sel_test = min(self.args.subset_size // 5, test_len)
                    dataset['test'] = dataset['test'].shuffle(seed=42).select(range(sel_test))
                print(f"  Using subset: {len(dataset['train'])} train, {len(dataset.get('test', []))} test")

        # Load tokenizer (use GPT-2 tokenizer by default for LM tasks)
        print(f"\n🔤 Loading tokenizer...")
        tokenizer_name = self.args.tokenizer or "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        print(f"  ✓ Tokenizer loaded (vocab: {len(tokenizer)})")

        return dataset, tokenizer, num_classes, use_streaming
    
    def benchmark(self):
        """Run comprehensive benchmarks"""
        print("\n" + "=" * 80)
        print("BENCHMARKING MODE")
        print("=" * 80)
        
        results = {}
        
        # Test different configurations
        test_configs = ["tiny", "small", "medium", "large"]
        sequence_lengths = [512, 1024, 2048, 4096, 8192]
        
        for config_name in test_configs:
            print(f"\n📊 Testing {config_name.upper()} configuration...")
            config = CONFIGS[config_name]
            
            # Create model
            model_config = TemporalEigenstateConfig(
                dim=config['d_model'],
                n_layers=config['n_layers'],
                num_eigenstates=config['num_eigenstates'],
                max_seq_len=max(sequence_lengths),
                vocab_size=50000,
            )
            model = TemporalEigenstateNetwork(model_config).to(self.device)
            model.eval()
            
            config_results = {
                "parameters": sum(p.numel() for p in model.parameters()),
                "sequence_lengths": {}
            }
            
            # Test each sequence length
            for seq_len in sequence_lengths:
                if seq_len > config['max_seq_len']:
                    continue
                
                try:
                    batch_size = config['batch_size']
                    input_ids = torch.randint(0, 50000, (batch_size, seq_len)).to(self.device)
                    
                    # Warmup
                    with torch.no_grad():
                        for _ in range(3):
                            _ = model(input_ids)
                    
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    
                    # Benchmark
                    times = []
                    with torch.no_grad():
                        for _ in range(10):
                            start = time.time()
                            _ = model(input_ids)
                            torch.cuda.synchronize()
                            times.append(time.time() - start)
                    
                    avg_time = sum(times) / len(times)
                    throughput = batch_size * seq_len / avg_time
                    memory = torch.cuda.max_memory_allocated() / 1024**3
                    
                    config_results["sequence_lengths"][seq_len] = {
                        "time_ms": avg_time * 1000,
                        "throughput": throughput,
                        "memory_gb": memory,
                        "batch_size": batch_size
                    }
                    
                    print(f"  {seq_len:5d} tokens: {avg_time*1000:6.2f}ms | {throughput:8.0f} tok/s | {memory:5.2f}GB")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"  {seq_len:5d} tokens: OOM")
                        torch.cuda.empty_cache()
                    else:
                        raise
            
            results[config_name] = config_results
            
            del model
            torch.cuda.empty_cache()
        
        # Save results
        output_file = self.output_dir / "benchmark_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Benchmark results saved to {output_file}")
        self.print_cost()
        
        return results
    
    def train(self):
        """Main training loop"""
        # Create model
        model = self.create_model()
        
        # Track if using streaming (for later logic)
        use_streaming = False
        
        # Try to compile model for extra speed (PyTorch 2.x)
        # torch.compile() with proper backend selection for L40S/Ada
        if not self.args.no_compile:
            try:
                print(f"\n🔥 Compiling model with torch.compile() for 2-3× speedup...")
                # Use 'reduce-overhead' mode for recurrent models like TEN
                # This optimizes for repeated small operations (our eigenstate evolution)
                model = torch.compile(
                    model, 
                    mode='reduce-overhead',  # Better for recurrent/sequential ops
                    fullgraph=False,  # Allow graph breaks for flexibility
                    dynamic=False  # Static shapes for max speed
                )
                print(f"  ✓ Model compiled successfully with 'reduce-overhead' mode!")
                print(f"  ⚡ Expected 2-3× speedup from kernel fusion and TF32")
            except Exception as e:
                print(f"  ⚠️  torch.compile() failed: {e}")
                print(f"  Continuing without compilation...")
        else:
            print(f"\n⚠️  Skipping torch.compile() (--no_compile flag set)")
        
        # Check if using pre-tokenized data
        if self.args.pretokenized:
            print(f"\n⚡ FAST MODE: Using pre-tokenized data!")
            tokenized_dir = Path(self.args.tokenized_dir or f"/root/ten_workspace/tokenized/{self.args.dataset}")
            
            if not tokenized_dir.exists():
                print(f"\n❌ ERROR: Pre-tokenized data not found at {tokenized_dir}")
                print(f"\nRun this first:")
                print(f"  python3 scripts/pretokenize_and_pack.py --dataset {self.args.dataset} --chunk_size {self.config['max_seq_len']}")
                return
            
            # Load pre-tokenized dataset with aggressive caching
            # Cache size: 256 chunks × 32K tokens × 2 bytes = ~16MB per chunk × 256 = ~4GB cache
            # Pass max_seq_len to automatically truncate chunks if needed
            dataset = PreTokenizedDataset(
                tokenized_dir, 
                cache_size=256,
                max_seq_len=self.config['max_seq_len']
            )
            
            # Load metadata for tokenizer info
            with open(tokenized_dir / "metadata.json") as f:
                metadata = json.load(f)
            
            # Create simple tokenizer wrapper for vocab info
            class TokenizerMock:
                def __init__(self, metadata):
                    self.pad_token_id = metadata.get('pad_token_id')
                    self.eos_token_id = metadata.get('eos_token_id')
            
            tokenizer = TokenizerMock(metadata)
            
            # Create optimized DataLoader with aggressive prefetching
            # CRITICAL: Use 'spawn' instead of 'fork' to avoid deadlocks with torch.compile()
            # Fork can cause deadlocks when combined with CUDA and compiled models
            train_loader = DataLoader(
                dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True,
                persistent_workers=True if self.args.num_workers > 0 else False,
                prefetch_factor=4 if self.args.num_workers > 0 else None,  # Reduced from 16 to avoid memory pressure
                multiprocessing_context='spawn' if self.args.num_workers > 0 else None  # Changed from 'fork' to 'spawn'!
            )
            
            print(f"  ✓ Pre-tokenized data loaded: {len(dataset):,} chunks")
            print(f"  ✓ Fast DataLoader configured ({self.args.num_workers} workers, prefetch={4})")
            print(f"  ✓ Using 'spawn' multiprocessing (prevents deadlocks with torch.compile)")
            print(f"  ✓ Prefetching: {self.args.num_workers * 4} batches in pipeline!")
            
        else:
            # Standard tokenization path (slower)
            print(f"\n⚠️  SLOW MODE: Tokenizing during training")
            print(f"  Consider pre-tokenizing for 5-50× speedup:")
            print(f"    python3 scripts/pretokenize_and_pack.py --dataset {self.args.dataset}")
            
            dataset, tokenizer, num_classes, use_streaming = self.prepare_data()
        
            # If dry run requested, run a quick forward pass and exit
            if self.args.dry_run:
                print("\n⚡ Dry run: running a single forward pass with a small batch...")

                # Find a textual field in the dataset examples
                sample_example = dataset['train'][0]
                text_field = None
                for k, v in sample_example.items():
                    if isinstance(v, str):
                        text_field = k
                        break

                if text_field is None:
                    raise RuntimeError("No textual field found in dataset for dry run")

                n = max(1, min(self.args.dry_samples, len(dataset['train'])))
                texts = [dataset['train'][i][text_field] for i in range(n)]

                # Tokenize to model max length
                max_len = self.config.get('max_seq_len', 1024)
                tokenized = tokenizer(texts, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
                input_ids = tokenized['input_ids'].to(self.device)

                model.eval()
                with torch.no_grad():
                    out = model(input_ids)

                print(f"  ✓ Dry run forward pass successful: input {input_ids.shape} -> output {out.shape}")
                print(f"  GPU memory (GB): {torch.cuda.max_memory_allocated() / 1024**3:.2f}")
                return
            
            # Prepare tokenized dataset for language modeling
            def tokenize_function(examples):
                # Find text field
                text_field = 'text'
                if text_field not in examples:
                    # Try other common field names
                    for field in ['content', 'article', 'document']:
                        if field in examples:
                            text_field = field
                            break
                
                return tokenizer(
                    examples[text_field],
                    truncation=True,
                    max_length=self.config['max_seq_len'],
                    padding='max_length',
                    return_tensors='pt'
                )
            
            print("\n📝 Tokenizing dataset...")
            
            # Handle streaming vs regular datasets
            if use_streaming:
                # For streaming datasets, don't call .map() with remove_columns
                # Just tokenize on-the-fly in the DataLoader collate_fn
                print("  🌊 Streaming mode: tokenizing on-the-fly during training")
                tokenized_train = dataset['train']
                
                # Create a custom collate function for streaming
                def collate_fn(batch):
                    # Find text field
                    text_field = 'text'
                    if text_field not in batch[0]:
                        for field in ['content', 'article', 'document']:
                            if field in batch[0]:
                                text_field = field
                                break
                    
                    texts = [item[text_field] for item in batch]
                    tokenized = tokenizer(
                        texts,
                        truncation=True,
                        max_length=self.config['max_seq_len'],
                        padding='max_length',
                        return_tensors='pt'
                    )
                    return tokenized
                
            else:
                # Regular dataset: tokenize everything upfront
                tokenized_train = dataset['train'].map(
                    tokenize_function,
                    batched=True,
                    remove_columns=dataset['train'].column_names,
                    desc="Tokenizing train"
                )
                tokenized_train.set_format('torch')
                collate_fn = None
            
            # Create data loader
            train_loader = DataLoader(
                tokenized_train,
                batch_size=self.config['batch_size'],
                shuffle=not use_streaming,  # Can't shuffle streaming datasets
                num_workers=self.args.num_workers if not use_streaming else 0,  # No workers for streaming
                pin_memory=True,
                persistent_workers=(True if self.args.num_workers > 0 and not use_streaming else False),
                collate_fn=collate_fn
            )
            
            if use_streaming:
                print(f"  ✓ Streaming data loader ready (tokenizing on-the-fly)")
            else:
                print(f"  ✓ Data loader ready ({len(train_loader)} batches per epoch)")
        
        # Setup training
        print(f"\n⚙️  Setting up training...")
        
        # Enable gradient checkpointing if requested
        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            print(f"  ✓ Gradient checkpointing enabled (trades compute for memory)")
        
        # Use 8-bit optimizer if requested for memory efficiency
        if self.args.use_8bit_optim:
            try:
                from bitsandbytes.optim import AdamW8bit
                optimizer = AdamW8bit(
                    model.parameters(),
                    lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay,
                    betas=(0.9, 0.999)
                )
                print(f"  ✓ Using 8-bit AdamW optimizer (bitsandbytes)")
            except ImportError:
                print(f"  ⚠️  bitsandbytes not installed, using standard AdamW")
                print(f"     Install with: pip install bitsandbytes")
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay,
                    betas=(0.9, 0.999)
                )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                betas=(0.9, 0.999)
            )
        
        # Mixed precision
        use_amp = self.args.mixed_precision
        scaler = GradScaler('cuda') if use_amp else None
        print(f"  Mixed precision: {'Enabled' if use_amp else 'Disabled'}")
        
        print(f"  Optimizer: {'AdamW8bit' if self.args.use_8bit_optim else 'AdamW'} (lr={self.args.learning_rate})")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Gradient accumulation: {self.args.gradient_accumulation}")
        print(f"  DataLoader workers: {self.args.num_workers if not (hasattr(self, 'use_streaming') and use_streaming) else 0}")
        print(f"  Epochs: {self.args.epochs}")
        
        # Calculate total steps (can't do this for streaming datasets)
        if not (self.args.pretokenized or use_streaming):
            total_steps = len(train_loader) * self.args.epochs
            print(f"  Total steps: {total_steps:,}")
            print(f"  Batches per epoch: {len(train_loader):,}")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
        else:
            # For streaming/pretokenized, use a large estimate
            total_steps = 100000  # Estimate
            print(f"  Total steps: streaming (unknown)")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
        
        # Training info
        print(f"\n🚀 Starting training...")
        print(f"  Expected time: {CONFIGS[self.args.config]['description']}")
        self.print_cost()
        
        # Training loop
        model.train()
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(self.args.epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{self.args.epochs}")
            print(f"{'='*80}")
            
            epoch_loss = 0
            optimizer.zero_grad()
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Memory tracking for first batch
                if batch_idx == 0:
                    torch.cuda.reset_peak_memory_stats()
                
                # Handle both pre-tokenized and regular data
                if self.args.pretokenized:
                    # Pre-tokenized: batch is already input_ids tensor
                    input_ids = batch.to(self.device)
                else:
                    # Regular: batch is dict with 'input_ids' key
                    input_ids = batch['input_ids'].to(self.device)
                
                # Create labels for language modeling (predict next token)
                # Shift by one: input=[0,1,2,3], label=[1,2,3,PAD]
                labels = input_ids[:, 1:].contiguous()
                inputs = input_ids[:, :-1].contiguous()
                
                # Memory checkpoint: after data load
                if batch_idx == 0:
                    mem_data = torch.cuda.max_memory_allocated() / 1024**3
                    print(f"\n  Memory after data load: {mem_data:.2f}GB")
                
                # Forward pass with new compute_loss API (includes energy regularization)
                if use_amp:
                    with autocast(device_type='cuda', dtype=torch.float16):
                        # Memory checkpoint: after forward
                        if batch_idx == 0:
                            mem_fwd_start = torch.cuda.max_memory_allocated() / 1024**3
                        
                        # Use model.compute_loss() for paper-compliant training with energy regularization
                        # Note: loss_dict['loss'] = ce_loss + energy_reg_weight * energy_loss (already weighted!)
                        loss_dict = model.compute_loss(inputs, labels, return_dict=True)
                        loss = loss_dict['loss'] / self.args.gradient_accumulation
                        
                        # Track loss components for monitoring (raw energy loss before weighting)
                        ce_loss_value = loss_dict['ce_loss'].item()
                        energy_loss_value = loss_dict['energy_loss'].item()
                        
                        # Memory checkpoint: after loss
                        if batch_idx == 0:
                            mem_fwd = torch.cuda.max_memory_allocated() / 1024**3
                            print(f"  Memory after forward+loss: {mem_fwd:.2f}GB (delta: {mem_fwd-mem_data:.2f}GB)")
                            print(f"  Loss breakdown - CE: {ce_loss_value:.4f}, Energy (raw): {energy_loss_value:.4f}")
                    
                    # Memory checkpoint: before backward
                    if batch_idx == 0:
                        mem_loss = torch.cuda.max_memory_allocated() / 1024**3
                    
                    scaler.scale(loss).backward()
                    
                    # Memory checkpoint: after backward
                    if batch_idx == 0:
                        mem_bwd = torch.cuda.max_memory_allocated() / 1024**3
                        print(f"  Memory after backward: {mem_bwd:.2f}GB (delta: {mem_bwd-mem_loss:.2f}GB)")
                        print(f"  TOTAL PEAK: {mem_bwd:.2f}GB\n")
                else:
                    # Use model.compute_loss() for paper-compliant training with energy regularization
                    loss_dict = model.compute_loss(inputs, labels, return_dict=True)
                    loss = loss_dict['loss'] / self.args.gradient_accumulation
                    
                    # Track loss components for monitoring (raw energy loss before weighting)
                    ce_loss_value = loss_dict['ce_loss'].item()
                    energy_loss_value = loss_dict['energy_loss'].item()
                    
                    loss.backward()
                
                # Store loss values before cleanup
                loss_value = loss.item() * self.args.gradient_accumulation
                epoch_loss += loss_value
                
                # CRITICAL: Aggressive memory cleanup every batch
                # Delete intermediate tensors explicitly
                del loss_dict, loss, labels, inputs, input_ids
                
                # Gradient accumulation
                if (batch_idx + 1) % self.args.gradient_accumulation == 0:
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # Force garbage collection every step
                    if global_step % 10 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                
                # Update progress bar with loss breakdown
                progress_bar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'ce': f'{ce_loss_value:.4f}',
                    'energy': f'{energy_loss_value:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                
                # Save checkpoint
                if global_step > 0 and global_step % self.args.save_steps == 0:
                    checkpoint_path = self.output_dir / "checkpoints" / f"checkpoint-{global_step}.pt"
                    torch.save({
                        'step': global_step,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss.item(),
                    }, checkpoint_path)
                    print(f"\n💾 Checkpoint saved: {checkpoint_path}")
            
            # Epoch summary
            avg_loss = epoch_loss / len(train_loader)
            print(f"\n📊 Epoch {epoch + 1} Summary:")
            print(f"  Average loss: {avg_loss:.4f}")
            self.print_cost()
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = self.output_dir / "checkpoints" / "best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'loss': best_loss,
                    'config': self.config,
                }, best_path)
                print(f"  ⭐ New best model saved: {best_path}")
        
        # Final save
        final_path = self.output_dir / "checkpoints" / "final_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'total_steps': global_step,
        }, final_path)
        
        print(f"\n{'='*80}")
        print(f"✅ Training Complete!")
        print(f"{'='*80}")
        print(f"  Total steps: {global_step}")
        print(f"  Final loss: {avg_loss:.4f}")
        print(f"  Best loss: {best_loss:.4f}")
        print(f"  Model saved: {final_path}")
        self.print_cost()


def main():
    """Main training function - must be called from __main__ for multiprocessing"""
    parser = argparse.ArgumentParser(description="Train TEN on DigitalOcean GPU")
    
    # Model configuration
    parser.add_argument("--config", type=str, default="medium",
                       choices=list(CONFIGS.keys()),
                       help="Model configuration preset")
    
    # Training parameters
    parser.add_argument("--dataset", type=str, default="imdb",
                       choices=["imdb", "ag_news", "wikitext-103", "tinystories", "finewebedu", "pg19", "openwebtext"],
                       help="Dataset to use")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--gradient_accumulation", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--mixed_precision", action="store_true",
                       help="Use mixed precision training (faster)")
    
    # Data parameters
    parser.add_argument("--vocab_size", type=int, default=50257,
                       help="Vocabulary size (50257 for GPT-2, 30522 for BERT)")
    parser.add_argument("--subset_size", type=int, default=0,
                       help="Use dataset subset (0 = full dataset)")
    parser.add_argument("--max_seq_len", type=int, default=None,
                       help="Override max sequence length")
    parser.add_argument("--save_steps", type=int, default=1000,
                       help="Save checkpoint every N steps")
    
    # Speed optimizations
    parser.add_argument("--pretokenized", action="store_true",
                       help="Use pre-tokenized data (5-50× faster!)")
    parser.add_argument("--tokenized_dir", type=str, default=None,
                       help="Directory with pre-tokenized chunks (default: /root/ten_workspace/tokenized/{dataset})")
    parser.add_argument("--streaming", action="store_true",
                       help="Use streaming mode (no full download, start training immediately)")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="DataLoader workers (default: 8, set to 0 to disable)")
    parser.add_argument("--use_8bit_optim", action="store_true",
                       help="Use 8-bit AdamW optimizer (requires bitsandbytes)")
    parser.add_argument("--no_compile", action="store_true",
                       help="Disable torch.compile() (use if training hangs/deadlocks)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing to reduce memory (trades compute for memory)")
    
    # Paper-compliant features
    parser.add_argument("--use_hten", action="store_true",
                       help="Enable Hierarchical TEN (multi-scale processing, Section 5)")
    parser.add_argument("--hten_scales", type=str, default=None,
                       help="HTEN scales as comma-separated list (e.g., '1,2,4,8')")
    parser.add_argument("--energy_reg_weight", type=float, default=None,
                       help="Energy regularization weight (Theorem 4, default: 0.01)")

    # Dry-run and tokenizer options
    parser.add_argument("--dry_run", action="store_true",
                       help="Quickly build model and run one forward pass using a small batch")
    parser.add_argument("--dry_samples", type=int, default=2,
                       help="Number of samples to use for --dry_run")
    parser.add_argument("--tokenizer", type=str, default=None,
                       help="Tokenizer name to use (e.g. gpt2, bert-base-uncased). Default: gpt2")
    
    # Output
    parser.add_argument("--output_dir", type=str, 
                       default=os.path.expanduser("~/ten_workspace"),
                       help="Output directory")
    
    # Special modes
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark mode instead of training")
    
    args = parser.parse_args()
    
    # Get configuration
    config = CONFIGS[args.config].copy()
    if args.max_seq_len:
        config['max_seq_len'] = args.max_seq_len
    
    # Override with command-line arguments for paper features
    if args.use_hten:
        config['use_hten'] = True
    if args.hten_scales:
        config['hten_scales'] = [int(x.strip()) for x in args.hten_scales.split(',')]
    if args.energy_reg_weight is not None:
        config['energy_reg_weight'] = args.energy_reg_weight
    
    # Print configuration
    print("\n" + "=" * 80)
    print(f"DigitalOcean L40S/RTX 6000 Ada Training")
    print("=" * 80)
    print(f"\nConfiguration: {args.config.upper()}")
    print(f"Description: {config['description']}")
    print(f"Parameters:")
    for key, value in config.items():
        if key != 'description':
            print(f"  {key}: {value}")
    print("\nGPU: 48GB VRAM (L40S or RTX 6000 Ada)")
    print("Cost: $1.57/hour")
    print("Budget: $15 free credit (~9.5 hours)")
    print("=" * 80)
    
    # Create trainer
    trainer = DigitalOceanTrainer(config, args)
    
    # Run benchmark or training
    if args.benchmark:
        trainer.benchmark()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
