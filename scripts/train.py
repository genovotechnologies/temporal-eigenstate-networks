"""
Training script for TEN and baselines on WikiText-103.
Trains all models for equal wall-clock time and reports final perplexity.

Usage:
    python train.py --model ten --seq_len 2048 --hours 8 --wandb
    python train.py --model transformer_flash --seq_len 2048 --hours 8
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from datasets import load_dataset
from transformers import AutoTokenizer

from benchmark import MODEL_REGISTRY, evaluate_perplexity


class TokenDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n_chunks = len(tokens) // seq_len

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        return self.tokens[start:start + self.seq_len]


def get_datasets(seq_len):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    train_data = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    val_data = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation')
    test_data = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')

    def tokenize_split(data):
        text = '\n'.join([t for t in data['text'] if len(t) > 0])
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor(tokens, dtype=torch.long)

    train_tokens = tokenize_split(train_data)
    val_tokens = tokenize_split(val_data)
    test_tokens = tokenize_split(test_data)

    # +1 for target offset
    train_ds = TokenDataset(train_tokens, seq_len + 1)
    val_ds = TokenDataset(val_tokens, seq_len + 1)
    test_ds = TokenDataset(test_tokens, seq_len + 1)

    return train_ds, val_ds, test_ds, vocab_size


def train(args):
    device = torch.device(args.device)

    print(f"Loading data (seq_len={args.seq_len})...")
    train_ds, val_ds, test_ds, vocab_size = get_datasets(args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True, drop_last=True)

    print(f"Building {args.model} (vocab={vocab_size})...")
    build_fn = MODEL_REGISTRY[args.model]
    model = build_fn(vocab_size, d_model=args.d_model, n_layers=args.n_layers,
                     max_seq_len=args.seq_len + 1).to(device)

    n_params = model.count_parameters()
    print(f"Parameters: {n_params / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.999), weight_decay=0.1)

    total_steps = len(train_loader) * args.max_epochs
    warmup_steps = min(2000, total_steps // 10)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    scaler = GradScaler('cuda')

    max_time = args.hours * 3600
    start_time = time.time()
    global_step = 0
    best_val_ppl = float('inf')

    print(f"\nTraining for {args.hours}h or {args.max_epochs} epochs...")
    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    print(f"Precision: {'bf16' if args.bf16 else 'fp16'}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*60}")

    try:
        for epoch in range(args.max_epochs):
            model.train()
            epoch_loss = 0
            epoch_tokens = 0

            for batch_idx, batch in enumerate(train_loader):
                if time.time() - start_time > max_time:
                    print(f"\nTime limit reached ({args.hours}h)")
                    break

                input_ids = batch[:, :-1].to(device)
                targets = batch[:, 1:].to(device)

                dtype = torch.bfloat16 if args.bf16 else torch.float16
                with autocast('cuda', dtype=dtype):
                    logits = model(input_ids)
                    loss = nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1)
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                epoch_loss += loss.item() * targets.numel()
                epoch_tokens += targets.numel()
                global_step += 1

                if global_step % 100 == 0:
                    avg_loss = epoch_loss / epoch_tokens
                    ppl = math.exp(avg_loss)
                    elapsed = time.time() - start_time
                    tps = epoch_tokens / elapsed
                    lr = scheduler.get_last_lr()[0]
                    print(f"  step {global_step:6d} | loss {avg_loss:.4f} | ppl {ppl:.1f} | "
                          f"lr {lr:.2e} | {tps:.0f} tok/s | {elapsed/60:.1f}min")

            if time.time() - start_time > max_time:
                break

            # Validation
            val_ppl = evaluate_perplexity(model, val_loader, device, max_batches=50)
            print(f"\nEpoch {epoch+1} | Val PPL: {val_ppl:.2f}")

            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                torch.save(model.state_dict(), f'best_{args.model}.pt')
                print(f"  New best! Saved checkpoint.")

            model.train()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    # Final evaluation
    print(f"\n{'='*60}")
    print("Final evaluation on test set...")
    model.load_state_dict(torch.load(f'best_{args.model}.pt', weights_only=True))
    test_ppl = evaluate_perplexity(model, test_loader, device, max_batches=100)

    # Throughput measurement
    torch.cuda.reset_peak_memory_stats(device)
    sample = next(iter(test_loader))[:, :-1].to(device)
    from benchmark import measure_throughput
    perf = measure_throughput(model, sample)

    elapsed_total = time.time() - start_time

    results = {
        'model': args.model,
        'params_M': round(n_params / 1e6, 1),
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'seq_len': args.seq_len,
        'batch_size': args.batch_size,
        'precision': 'bf16' if args.bf16 else 'fp16',
        'test_perplexity': test_ppl,
        'best_val_perplexity': best_val_ppl,
        'training_time_hours': round(elapsed_total / 3600, 2),
        'total_steps': global_step,
        **perf,
        'gpu': torch.cuda.get_device_name(0),
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__,
    }

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS: {args.model}")
    print(f"{'='*60}")
    print(json.dumps(results, indent=2))

    output_file = f'results_{args.model}_{args.seq_len}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--hours', type=float, default=8.0)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    train(args)
