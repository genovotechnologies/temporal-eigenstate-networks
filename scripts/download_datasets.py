#!/usr/bin/env python3
"""
Download and Prepare Large-Scale Datasets for TEN Training
Optimized for long-range tasks and pretraining on DigitalOcean L40S/RTX 6000
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset
import argparse

# Add colors for output
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'

def print_status(msg):
    print(f"{GREEN}[✓]{NC} {msg}")

def print_info(msg):
    print(f"{BLUE}[ℹ]{NC} {msg}")

def print_warning(msg):
    print(f"{YELLOW}[⚠]{NC} {msg}")


DATASETS_INFO = {
    "wikitext-103": {
        "name": "wikitext",
        "config": "wikitext-103-raw-v1",
        "size": "~500MB",
        "samples": "~1.8M articles",
        "avg_length": "~3800 tokens",
        "best_for": "Language modeling pretraining",
        "task": "language_modeling",
        "download_time": "~2-3 min",
        "recommended": True,
        "description": "Wikipedia articles, perfect for pretraining. Long sequences."
    },
    "wikitext-2": {
        "name": "wikitext",
        "config": "wikitext-2-raw-v1",
        "size": "~12MB",
        "samples": "36K articles",
        "avg_length": "~3800 tokens",
        "best_for": "Quick testing of language modeling",
        "task": "language_modeling",
        "download_time": "~30 sec",
        "recommended": False,
        "description": "Smaller WikiText for testing."
    },
    "bookcorpus": {
        "name": "bookcorpusopen/bookcorpusopen",
        "config": None,
        "size": "~4.5GB",
        "samples": "~74M sentences from 11K books",
        "avg_length": "~150 tokens per sentence",
        "best_for": "Large-scale pretraining with books",
        "task": "language_modeling",
        "download_time": "~10-15 min",
        "recommended": True,
        "description": "Books corpus used to train BERT. Excellent for pretraining."
    },
    "openwebtext": {
        "name": "Skylion007/openwebtext",
        "config": None,
        "size": "~38GB",
        "samples": "~8M documents",
        "avg_length": "varies, many long documents",
        "best_for": "Large-scale web text pretraining",
        "task": "language_modeling",
        "download_time": "~30-60 min",
        "recommended": True,
        "description": "Open source recreation of WebText (used for GPT-2). Huge and diverse."
    },
    "c4": {
        "name": "c4",
        "config": "en",
        "size": "~300GB (can use subset)",
        "samples": "~365M documents",
        "avg_length": "varies widely",
        "best_for": "Massive-scale pretraining",
        "task": "language_modeling",
        "download_time": "Can stream or use subset",
        "recommended": False,  # Too large for quick experiments
        "description": "Colossal Clean Crawled Corpus. Massive web text."
    },
    "arxiv": {
        "name": "togethercomputer/RedPajama-Data-1T-Sample",
        "config": None,
        "size": "~1GB sample",
        "samples": "~1B tokens (mixed sources)",
        "avg_length": "varies, many long documents",
        "best_for": "Large-scale pretraining with long sequences",
        "task": "language_modeling",
        "download_time": "~3-5 min",
        "recommended": True,
        "description": "RedPajama 1T sample - high quality pretraining data including arXiv, books, code."
    },
    "pubmed": {
        "name": "EleutherAI/pile",
        "config": "all",
        "size": "~825GB (use subset)",
        "samples": "~210B tokens",
        "avg_length": "varies widely",
        "best_for": "Massive diverse pretraining",
        "task": "language_modeling",
        "download_time": "Can stream or use subset",
        "recommended": False,  # Too large
        "description": "The Pile - huge diverse dataset including books, papers, code, web."
    },
    "pg19": {
        "name": "emozilla/pg19-test",
        "config": None,
        "size": "~1GB (test subset)",
        "samples": "~100 books",
        "avg_length": "50K-100K tokens per book",
        "best_for": "Ultra-long range modeling",
        "task": "language_modeling",
        "download_time": "~3-5 min",
        "recommended": True,
        "description": "Project Gutenberg books subset. PERFECT for testing long-range (8K+ tokens)."
    },
    "squad": {
        "name": "squad",
        "config": None,
        "size": "~35MB",
        "samples": "~100K questions",
        "avg_length": "~150-500 tokens",
        "best_for": "Question answering",
        "task": "qa",
        "download_time": "~1 min",
        "recommended": False,  # Not for pretraining
        "description": "Stanford Question Answering Dataset."
    },
    "multirc": {
        "name": "super_glue",
        "config": "multirc",
        "size": "~10MB",
        "samples": "~10K questions",
        "avg_length": "~300-600 tokens",
        "best_for": "Multi-sentence reading comprehension",
        "task": "reading_comprehension",
        "download_time": "~30 sec",
        "recommended": False,
        "description": "Multi-sentence reasoning."
    },
    "tinystories": {
        "name": "roneneldan/TinyStories",
        "config": None,
        "size": "~2GB",
        "samples": "~2M short stories",
        "avg_length": "~200-500 tokens",
        "best_for": "Quick pretraining test",
        "task": "language_modeling",
        "download_time": "~5 min",
        "recommended": True,
        "description": "High quality synthetic stories - fast to train, good for testing."
    },
    "finewebedu": {
        "name": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "size": "~10B tokens (~40GB)",
        "samples": "~9.6M documents",
        "avg_length": "~1000 tokens",
        "best_for": "High quality web pretraining",
        "task": "language_modeling",
        "download_time": "~20-30 min for sample",
        "recommended": True,
        "description": "FineWeb-Edu - highest quality web corpus filtered for educational content."
    },
    "codeparrot": {
        "name": "codeparrot/github-code",
        "config": "default",
        "size": "~5GB",
        "samples": "~1M code files",
        "avg_length": "~500-2000 tokens",
        "best_for": "Code pretraining",
        "task": "language_modeling",
        "download_time": "~10-15 min",
        "recommended": False,
        "description": "GitHub code repository for code understanding tasks."
    },
}


def show_menu():
    """Show dataset selection menu"""
    print("\n" + "="*80)
    print("DATASET SELECTION FOR TEN TRAINING")
    print("="*80)
    print("\nRecommended datasets for your goals (long-range + pretraining):\n")
    
    # Show recommended first
    recommended = []
    others = []
    
    for i, (key, info) in enumerate(DATASETS_INFO.items(), 1):
        if info['recommended']:
            recommended.append((i, key, info))
        else:
            others.append((i, key, info))
    
    print("🌟 HIGHLY RECOMMENDED:")
    for i, key, info in recommended:
        print(f"\n{i}. {key.upper()}")
        print(f"   Size: {info['size']} | Samples: {info['samples']}")
        print(f"   Best for: {info['best_for']}")
        print(f"   Avg length: {info['avg_length']}")
        print(f"   Download: {info['download_time']}")
        print(f"   → {info['description']}")
    
    print("\n\n📚 OTHER OPTIONS:")
    for i, key, info in others:
        print(f"\n{i}. {key.upper()}")
        print(f"   Size: {info['size']} | Samples: {info['samples']}")
        print(f"   Best for: {info['best_for']}")
        print(f"   → {info['description']}")
    
    print("\n" + "="*80)


def download_dataset(dataset_key, output_dir="~/ten_workspace/data"):
    """Download and prepare a dataset"""
    
    if dataset_key not in DATASETS_INFO:
        print(f"Error: Unknown dataset '{dataset_key}'")
        return False
    
    info = DATASETS_INFO[dataset_key]
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"DOWNLOADING: {dataset_key.upper()}")
    print("="*80)
    print(f"Size: {info['size']}")
    print(f"Samples: {info['samples']}")
    print(f"Estimated time: {info['download_time']}")
    print("="*80 + "\n")
    
    try:
        print_info("Starting download...")
        
        if info['config']:
            dataset = load_dataset(info['name'], info['config'])
        else:
            dataset = load_dataset(info['name'])
        
        print_status("Download complete!")
        
        # Print dataset info
        print("\n" + "-"*80)
        print("DATASET INFORMATION:")
        print("-"*80)
        
        for split in dataset.keys():
            num_samples = len(dataset[split])
            print(f"  {split.upper()}: {num_samples:,} samples")
        
        # Show sample
        if 'train' in dataset:
            print("\n" + "-"*80)
            print("SAMPLE (first item):")
            print("-"*80)
            sample = dataset['train'][0]
            for key, value in sample.items():
                if isinstance(value, str):
                    display_value = value[:200] + "..." if len(value) > 200 else value
                    print(f"  {key}: {display_value}")
                else:
                    print(f"  {key}: {value}")
        
        # Save dataset info
        info_file = output_dir / f"{dataset_key}_info.txt"
        with open(info_file, 'w') as f:
            f.write(f"Dataset: {dataset_key}\n")
            f.write(f"Name: {info['name']}\n")
            if info['config']:
                f.write(f"Config: {info['config']}\n")
            f.write(f"Size: {info['size']}\n")
            f.write(f"Task: {info['task']}\n")
            f.write(f"Description: {info['description']}\n\n")
            f.write("Splits:\n")
            for split in dataset.keys():
                f.write(f"  {split}: {len(dataset[split]):,} samples\n")
        
        print(f"\n{GREEN}[✓]{NC} Dataset info saved to: {info_file}")
        print(f"{GREEN}[✓]{NC} Dataset cached and ready to use!")
        print(f"\n{BLUE}[ℹ]{NC} The dataset is cached by HuggingFace and will be loaded automatically during training.")
        
        return True
        
    except Exception as e:
        print(f"\n{YELLOW}[✗]{NC} Error downloading dataset: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download large-scale datasets for TEN pretraining"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to download (see menu for options)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="~/ten_workspace/data",
        help="Output directory (default: ~/ten_workspace/data)"
    )
    parser.add_argument(
        "--menu",
        action="store_true",
        help="Show dataset menu and exit"
    )
    parser.add_argument(
        "--all-recommended",
        action="store_true",
        help="Download all recommended datasets"
    )
    
    args = parser.parse_args()
    
    if args.menu or not args.dataset:
        show_menu()
        
        if not args.dataset:
            print("\nUsage examples:")
            print("  python3 download_datasets.py --dataset wikitext-103")
            print("  python3 download_datasets.py --dataset pg19")
            print("  python3 download_datasets.py --dataset arxiv")
            print("  python3 download_datasets.py --all-recommended")
        return
    
    if args.all_recommended:
        print("\n" + "="*80)
        print("DOWNLOADING ALL RECOMMENDED DATASETS")
        print("="*80)
        print("This will download: wikitext-103, bookcorpus, openwebtext, arxiv, pg19, tinystories, finewebedu")
        print("Total size: ~100GB")
        print("Total time: ~90-120 minutes")
        print("="*80)
        
        response = input("\nContinue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
        
        recommended = [k for k, v in DATASETS_INFO.items() if v['recommended']]
        for dataset_key in recommended:
            success = download_dataset(dataset_key, args.output)
            if not success:
                print_warning(f"Failed to download {dataset_key}, continuing...")
        
        print("\n" + "="*80)
        print("ALL DOWNLOADS COMPLETE!")
        print("="*80)
    else:
        download_dataset(args.dataset, args.output)


if __name__ == "__main__":
    main()
