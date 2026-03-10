"""
Unified training script for architecture invariance experiment.
Trains Transformer, LSTM, or MLP on Shakespeare char-level data,
tracking per-unit entropy throughout training.

Usage:
    python train.py --arch transformer --out results/transformer
    python train.py --arch lstm --out results/lstm
    python train.py --arch mlp --out results/mlp
"""

import os
import sys
import time
import argparse
import json
import math

import numpy as np
import torch

from models import get_model, count_parameters
from entropy_tracker import (
    EntropyTracker,
    compute_attention_entropy,
    compute_hidden_entropy,
)

# ============================================================================
# Config
# ============================================================================

DATA_DIR = os.path.join(os.path.expanduser('~'), 'clawd', 'projects', 'nanoGPT', 'data', 'shakespeare_char')

# Architecture-shared hyperparameters
N_LAYER = 6
N_EMBD = 128
N_HEAD = 4  # transformer only
BLOCK_SIZE = 128
BATCH_SIZE = 32
DROPOUT = 0.1
VOCAB_SIZE = 65  # shakespeare char-level

# Training
MAX_ITERS = 3000
EVAL_INTERVAL = 50
EVAL_ITERS = 20
LOG_INTERVAL = 10
LEARNING_RATE = 1e-3
MIN_LR = 1e-4
WARMUP_ITERS = 100
LR_DECAY_ITERS = 5000
WEIGHT_DECAY = 1e-1

# ============================================================================
# Data loading (from nanoGPT)
# ============================================================================

def get_batch(split, device):
    data_file = os.path.join(DATA_DIR, f'{split}.bin')
    data = np.memmap(data_file, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy((data[i:i+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, device):
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        batch_losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split, device)
            _, loss = model(x, y)
            batch_losses[k] = loss.item()
        losses[split] = batch_losses.mean().item()
    model.train()
    return losses


def get_lr(it):
    """Learning rate schedule with warmup and cosine decay."""
    if it < WARMUP_ITERS:
        return LEARNING_RATE * it / WARMUP_ITERS
    if it > LR_DECAY_ITERS:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


# ============================================================================
# Entropy extraction per architecture
# ============================================================================

def extract_entropy(model, arch, tracker):
    """Extract per-unit entropy from the model and update tracker."""
    if arch == 'transformer':
        attn_weights_list = model.get_attn_weights()
        for layer_idx, attn_w in enumerate(attn_weights_list):
            if attn_w is not None:
                head_entropy = compute_attention_entropy(attn_w)
                tracker.update(layer_idx, head_entropy)

    elif arch == 'lstm':
        hidden_states = model.get_hidden_states()
        for layer_idx, hs in enumerate(hidden_states):
            if hs is not None:
                unit_entropy = compute_hidden_entropy(hs)
                tracker.update(layer_idx, unit_entropy)

    elif arch == 'mlp':
        activations = model.get_activations()
        for layer_idx, act in enumerate(activations):
            if act is not None:
                unit_entropy = compute_hidden_entropy(act)
                tracker.update(layer_idx, unit_entropy)


# ============================================================================
# Main training loop
# ============================================================================

def train(arch, out_dir, device='cpu'):
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Architecture Invariance Experiment")
    print(f"Architecture: {arch}")
    print(f"Output: {out_dir}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Create model
    model_kwargs = dict(
        n_layer=N_LAYER,
        n_embd=N_EMBD,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT,
    )
    if arch == 'transformer':
        model_kwargs['n_head'] = N_HEAD
        n_units = N_HEAD
    else:
        n_units = N_EMBD

    model = get_model(arch, VOCAB_SIZE, **model_kwargs)
    model = model.to(device)

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    # Entropy tracker
    tracker = EntropyTracker(N_LAYER, n_units, device=device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    # Training log
    train_log = {
        'arch': arch,
        'n_params': n_params,
        'config': model_kwargs,
        'losses': [],
    }

    # Training loop
    best_val_loss = float('inf')
    t0 = time.time()

    for iter_num in range(MAX_ITERS):
        # Set learning rate
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Eval
        if iter_num % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, device)
            train_log['losses'].append({
                'step': iter_num,
                'train': losses['train'],
                'val': losses['val'],
            })

            # Extract entropy and log
            x, y = get_batch('train', device)
            model(x, y)
            extract_entropy(model, arch, tracker)
            tracker.log_step(iter_num)

            per_layer, overall = tracker.crystal_fraction()
            ordering, _ = tracker.layer_ordering()

            elapsed = time.time() - t0
            print(f"step {iter_num:5d} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | crystal {overall:.1%} | "
                  f"order {ordering} | lr {lr:.2e} | {elapsed:.1f}s")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']

        # Forward + backward
        x, y = get_batch('train', device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Extract entropy on training batches periodically
        if iter_num % LOG_INTERVAL == 0 and iter_num % EVAL_INTERVAL != 0:
            with torch.no_grad():
                extract_entropy(model, arch, tracker)

    # Final eval
    losses = estimate_loss(model, device)
    train_log['losses'].append({
        'step': MAX_ITERS,
        'train': losses['train'],
        'val': losses['val'],
    })
    tracker.log_step(MAX_ITERS)

    elapsed = time.time() - t0
    per_layer, overall = tracker.crystal_fraction()
    ordering, _ = tracker.layer_ordering()

    print(f"\n{'='*60}")
    print(f"Training complete in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Final crystal fraction: {overall:.1%}")
    print(f"Per-layer: {[f'{x:.1%}' for x in per_layer]}")
    print(f"Layer ordering (most→least crystallized): {ordering}")
    print(f"{'='*60}\n")

    # Save results
    tracker.save(os.path.join(out_dir, 'entropy.json'))
    train_log['final'] = {
        'best_val_loss': best_val_loss,
        'crystal_fraction': overall,
        'per_layer': per_layer,
        'ordering': ordering,
        'elapsed_seconds': elapsed,
    }
    with open(os.path.join(out_dir, 'train_log.json'), 'w') as f:
        json.dump(train_log, f, indent=2)

    print(f"Results saved to {out_dir}/")
    return train_log


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Architecture invariance experiment')
    parser.add_argument('--arch', type=str, required=True,
                        choices=['transformer', 'lstm', 'mlp'],
                        help='Model architecture')
    parser.add_argument('--out', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu, cuda, mps)')
    parser.add_argument('--iters', type=int, default=None,
                        help='Override max iterations')
    args = parser.parse_args()

    if args.iters:
        MAX_ITERS = args.iters
        LR_DECAY_ITERS = args.iters

    train(args.arch, args.out, args.device)
